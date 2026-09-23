import argparse
import asyncio
import concurrent.futures
import json
import os
import random
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pypdf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from syntok.segmenter import process
from tqdm import tqdm

from olmocr.bench.tests import TableTest, TestType, parse_html_tables
from olmocr.data.renderpdf import (
    get_png_dimensions_from_base64,
    render_pdf_to_base64png,
)


def download_s3_pdf(s3_path, local_path):
    """Download a PDF from S3 to a local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(["aws", "s3", "cp", s3_path, local_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0


def extract_code_block(initial_response):
    # Use regex to find the last instance of a code block
    # First try to find HTML specific code blocks
    html_blocks = re.findall(r"```html\n(.*?)```", initial_response, re.DOTALL)

    # If HTML blocks found, return the last one
    if html_blocks:
        return html_blocks[-1].strip()

    # Otherwise, try to find any code blocks
    code_blocks = re.findall(r"```\n(.*?)```", initial_response, re.DOTALL)

    # If code blocks found, return the last one
    if code_blocks:
        return code_blocks[-1].strip()

    # If no code blocks found with newlines after backticks, try without newlines
    html_blocks_no_newline = re.findall(r"```html(.*?)```", initial_response, re.DOTALL)
    if html_blocks_no_newline:
        return html_blocks_no_newline[-1].strip()

    code_blocks_no_newline = re.findall(r"```(.*?)```", initial_response, re.DOTALL)
    if code_blocks_no_newline:
        return code_blocks_no_newline[-1].strip()

    # Return empty string if no code blocks found
    return None


def generate_html_from_image(client, image_base64):
    """Call Claude API to generate HTML from an image using a multi-step prompting strategy."""
    png_width, png_height = get_png_dimensions_from_base64(image_base64)

    try:
        # Step 1: Initial analysis and column detection
        analysis_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": "Analyze this document and provide a detailed assessment of its structure. Focus specifically on:\n"
                            "1. How many columns does the document have? Is it single-column, two-column, three-column, or a mixed layout?\n"
                            "2. What are the main sections and content types (headings, paragraphs, lists, tables, images, etc.)?\n"
                            "3. Does it have headers, footers, page numbers, or other special elements?\n"
                            "4. Is there any complex formatting that would be challenging to reproduce in HTML?\n\n"
                            "Please be very precise about the number of columns and how they're arranged.",
                        },
                    ],
                }
            ],
        )

        analysis_text = ""
        for content in analysis_response.content:
            if content.type == "text":
                analysis_text += content.text

        # Step 2: Initial HTML generation with detailed layout instructions
        initial_response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=6000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}},
                        {
                            "type": "text",
                            "text": "Render this document as clean, semantic HTML. Here's my analysis of the document structure:\n\n"
                            f"{analysis_text}\n\n"
                            "Important requirements:\n"
                            "1. Use appropriate HTML tags for elements like headings, paragraphs, lists, tables, etc.\n"
                            "2. Use the <header> and <footer> tags to represent content at the top/bottom which would not normally be part of the main content, such as page numbers, etc.\n"
                            "3. Use a placeholder <div> tag with class 'image' which will render as a grey box with black outline to make sure images have their original size, shape, and position on the page.\n"
                            "4. Render any math equations and Latex inline using either \\[ \\] or \\( \\) delimeters.\n"
                            "5. CRITICAL: If the document has a multi-column layout, you MUST preserve the exact same number of columns in your HTML. Use CSS flexbox or grid to create the columns.\n"
                            "6. Focus on creating valid, accessible HTML that preserves the appearance and formatting of the original page as closely as possible.\n"
                            f"7. The webpage will be viewed with a fixed viewport size of {png_width // 2} pixels wide by {png_height // 2} pixels tall.\n\n"
                            "8. For multi-column layouts, use explicit CSS. The most important aspect is preserving the column structure of the original document - this is critical.\n\n"
                            "Enclose your HTML in a ```html code block.",
                        },
                    ],
                }
            ],
        )

        # Extract initial HTML
        initial_html = ""
        for content in initial_response.content:
            if content.type == "text":
                initial_html += content.text

        return extract_code_block(initial_html)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return None


def extract_page_from_pdf(input_path, output_path, page_num):
    """
    Extract a specific page from a PDF and save it as a new PDF.

    Args:
        input_path: Path to the input PDF
        output_path: Path to save the extracted page
        page_num: The page number to extract (1-indexed, converted to 0-indexed for pypdf)

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Read the input PDF
        reader = pypdf.PdfReader(input_path)

        # Convert to 0-indexed for pypdf
        zero_idx_page = page_num - 1

        # Check if page number is valid
        if zero_idx_page >= len(reader.pages) or zero_idx_page < 0:
            print(f"Page number {page_num} out of range for {input_path} with {len(reader.pages)} pages")
            return False

        # Create a new PDF with just the selected page
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[zero_idx_page])

        # Write the output PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        return True
    except Exception as e:
        print(f"Error extracting page {page_num} from {input_path}: {str(e)}")
        return False


async def render_pdf_with_playwright(html_content, output_pdf_path, png_width, png_height):
    """
    Render HTML content using Playwright and save it as PDF.
    Try different scale factors if needed to ensure the output is exactly one page.

    Args:
        html_content: HTML content to render
        output_pdf_path: Path to save the rendered PDF
        png_width: Width of the viewport
        png_height: Height of the viewport

    Returns:
        bool: True if rendering was successful with exactly one page, False otherwise
    """
    scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Try these scale factors in order

    for scale in scale_factors:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page(viewport={"width": int(png_width // 2 * scale), "height": int(png_height // 2 * scale)})

                # Set the HTML content
                await page.set_content(html_content)

                # Add in katex and setup auto rendering
                katex_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "katex")
                katex_css_path = os.path.join(katex_dir, "katex.min.css")
                katex_js_path = os.path.join(katex_dir, "katex.min.js")
                katex_autorender_js_path = os.path.join(katex_dir, "auto-render.min.js")

                await page.add_style_tag(path=katex_css_path)
                await page.add_script_tag(path=katex_js_path)
                await page.add_script_tag(path=katex_autorender_js_path)

                # Run the KaTeX auto-renderer immediately rather than waiting for DOMContentLoaded
                await page.evaluate("""
                    renderMathInElement(document.body, {
                        // customised options
                        // • auto-render specific keys, e.g.:
                        delimiters: [
                            {left: '\\\\(', right: '\\\\)', display: false},
                            {left: '\\\\[', right: '\\\\]', display: true}
                        ],
                        // • rendering keys, e.g.:
                        throwOnError: false
                    });
                """)

                # Save as PDF with formatting options
                await page.pdf(
                    path=output_pdf_path,
                    scale=scale,
                    print_background=True,
                )

                await browser.close()

                # Check if the output PDF has exactly one page
                try:
                    reader = pypdf.PdfReader(output_pdf_path)
                    if len(reader.pages) == 1:
                        print(f"Successfully rendered as a single page PDF with scale factor {scale}")
                        return True
                    else:
                        print(f"PDF has {len(reader.pages)} pages with scale factor {scale}, trying a smaller scale...")
                        # Continue to the next scale factor
                except Exception as pdf_check_error:
                    print(f"Error checking PDF page count: {pdf_check_error}")
                    return False

        except Exception as e:
            print(f"Error rendering PDF with Playwright at scale {scale}: {str(e)}")
            # Try the next scale factor

    print("Failed to render PDF as a single page with any scale factor")
    return False


def generate_tests_from_html(html_content: str, pdf_id: str, page_num: int, verbose_table_testing: bool = False) -> List[Dict]:
    """
    Generate tests from HTML content parsed from the PDF.

    Args:
        html_content: The HTML content of the page
        pdf_id: The unique identifier for the PDF
        page_num: The page number
        verbose_table_testing: Whether to print table test verification details

    Returns:
        A list of test dictionaries that can be saved as JSONL
    """

    # Helper function to convert superscripts and subscripts to Unicode
    def convert_superscripts_subscripts(element):
        # Map for superscript characters
        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "+": "⁺",
            "-": "⁻",
            "=": "⁼",
            "(": "⁽",
            ")": "⁾",
            "n": "ⁿ",
            "i": "ⁱ",
        }

        # Map for subscript characters
        subscript_map = {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₄",
            "5": "₅",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
            "+": "₊",
            "-": "₋",
            "=": "₌",
            "(": "₍",
            ")": "₎",
            "a": "ₐ",
            "e": "ₑ",
            "o": "ₒ",
            "x": "ₓ",
            "h": "ₕ",
            "k": "ₖ",
            "l": "ₗ",
            "m": "ₘ",
            "n": "ₙ",
            "p": "ₚ",
            "s": "ₛ",
            "t": "ₜ",
        }

        # Process all superscript tags
        for sup in element.find_all("sup"):
            sup_text = sup.get_text()
            unicode_text = ""
            for char in sup_text:
                unicode_text += superscript_map.get(char, char)
            sup.replace_with(unicode_text)

        # Process all subscript tags
        for sub in element.find_all("sub"):
            sub_text = sub.get_text()
            unicode_text = ""
            for char in sub_text:
                unicode_text += subscript_map.get(char, char)
            sub.replace_with(unicode_text)

        return element

    tests = []
    pdf_filename = f"{pdf_id}_page{page_num}.pdf"
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove any divs or spans with class "line-number"
    for element in soup.find_all(["div", "span"], class_="line-number"):
        element.extract()

    # Rewrite any page-header and page-footer divs to be normalized to headers
    # Convert div.page-footer to footer in one line
    for div in soup.find_all("div", class_="page-header"):
        div.name = "header"

    for div in soup.find_all("div", class_="page-footer"):
        div.name = "footer"

    # Remove elements in the body that appear before the header or after the footer
    body = soup.find("body")
    if body:
        header = soup.find("header")
        footer = soup.find("footer")

        if header:
            # Remove elements before the header
            current = body.contents[0]
            while current and current != header:
                next_elem = current.next_sibling
                current.extract()
                current = next_elem

        if footer:
            # Remove elements after the footer
            current = footer.next_sibling
            while current:
                next_elem = current.next_sibling
                current.extract()
                current = next_elem

    # Step 1: Process headers, footers, and page numbers for TextAbsenceTests
    headers = soup.find_all("header")
    footers = soup.find_all("footer")
    page_numbers = soup.find_all("div", class_="page-number")

    # Function to create absence tests from text elements
    def create_absence_tests_from_elements(parent_element, element_type):
        mini_soup = BeautifulSoup(str(parent_element), "html.parser")

        # Convert superscripts and subscripts in the mini soup
        convert_superscripts_subscripts(mini_soup)

        # Remove headers, footers, and tables from the main_soup
        for element in mini_soup.find_all(["h1", "h2"]):
            element.extract()

        # Find all text-containing leaf elements within the parent
        text_elements = []

        # Get all target elements
        target_tags = mini_soup.find_all(["span", "div", "p", "h3", "h4", "h5", "h6"])

        # Filter to only include leaf nodes (elements that don't contain other target elements)
        for tag in target_tags:
            # Check if this element has no children from our target tags
            is_leaf = not tag.find(["span", "div", "p", "h3", "h4", "h5", "h6"])

            if is_leaf:
                text = tag.get_text().strip()
                if text:
                    text_elements.append(text)

        # If no elements found, use the parent's text as a fallback, but only if
        if not text_elements:
            parent_text = mini_soup.get_text().strip()
            if parent_text:
                text_elements.append(parent_text)

        # Create tests for each text element
        for text in text_elements:
            if "\n" in text:
                text = text.split("\n")[0]

            if len(text) > 3 or len([c for c in text if c.isdigit()]):  # Only create tests for meaningful text
                tests.append(
                    {
                        "pdf": pdf_filename,
                        "page": page_num,
                        "id": f"{pdf_id}_{element_type}_{uuid.uuid4().hex[:8]}",
                        "type": TestType.ABSENT.value,
                        "text": text,
                        "max_diffs": round(len(text) * 0.05),
                    }
                )

    # Create TextAbsenceTests for headers
    for header in headers:
        create_absence_tests_from_elements(header, "header")

    # Create TextAbsenceTests for footers
    for footer in footers:
        create_absence_tests_from_elements(footer, "footer")

    # Create TextAbsenceTests for page numbers
    for page_number in page_numbers:
        # Convert any superscripts/subscripts in the page number
        page_number_soup = BeautifulSoup(str(page_number), "html.parser")
        convert_superscripts_subscripts(page_number_soup)
        page_number_text = page_number_soup.get_text().strip()

        if page_number_text:
            tests.append(
                {
                    "pdf": pdf_filename,
                    "page": page_num,
                    "id": f"{pdf_id}_page_number_{uuid.uuid4().hex[:8]}",
                    "type": TestType.ABSENT.value,
                    "text": page_number_text,
                    "max_diffs": 0,
                }
            )

    # Step 2: Generate tests from tables using parse_html_tables
    # Convert superscripts and subscripts to Unicode equivalents in tables
    table_soup = BeautifulSoup(html_content, "html.parser")

    # Convert superscripts and subscripts in the table HTML
    convert_superscripts_subscripts(table_soup)
    html_content_with_unicode = str(table_soup)

    table_data_list = parse_html_tables(html_content_with_unicode)

    for table_idx, table_data in enumerate(table_data_list):
        # Get the table data as a numpy array
        table_array = table_data.data

        # Skip tables that are too small
        if table_array.shape[0] < 2 or table_array.shape[1] < 2:
            continue

        # Get a limited number of cells to create tests for
        # Select random rows and columns, excluding header rows/columns
        non_header_rows = [i for i in range(table_array.shape[0]) if i not in table_data.header_rows]
        non_header_cols = [j for j in range(table_array.shape[1]) if j not in table_data.header_cols]

        # If we don't have enough non-header cells, use all cells
        if len(non_header_rows) < 2 or len(non_header_cols) < 2:
            cell_positions = [(i, j) for i in range(table_array.shape[0]) for j in range(table_array.shape[1])]
        else:
            cell_positions = [
                (i, j)
                for i in random.sample(non_header_rows, min(3, len(non_header_rows)))
                for j in random.sample(non_header_cols, min(2, len(non_header_cols)))
            ]

        random.shuffle(cell_positions)

        # Create tests for each selected cell
        for row_idx, col_idx in cell_positions:
            cell_text = str(table_array[row_idx, col_idx]).strip()

            # Skip cells with minimal text
            if not cell_text or len(cell_text) < 3:
                continue

            # Create a TableTest with relevant relationships
            test_data = {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_table{table_idx}_{uuid.uuid4().hex[:8]}",
                "type": TestType.TABLE.value,
                "cell": cell_text,
                "max_diffs": 0,
            }

            # Check cell up
            if row_idx > 0:
                up_text = str(table_array[row_idx - 1, col_idx]).strip()
                if up_text and "\n" not in up_text:
                    test_data["up"] = up_text

            # Check cell down
            if row_idx < table_array.shape[0] - 1:
                down_text = str(table_array[row_idx + 1, col_idx]).strip()
                if down_text and "\n" not in down_text:
                    test_data["down"] = down_text

            # Check cell left
            if col_idx > 0:
                left_text = str(table_array[row_idx, col_idx - 1]).strip()
                if left_text and "\n" not in left_text:
                    test_data["left"] = left_text

            # Check cell right
            if col_idx < table_array.shape[1] - 1:
                right_text = str(table_array[row_idx, col_idx + 1]).strip()
                if right_text and "\n" not in right_text:
                    test_data["right"] = right_text

            # Check if current cell is a heading cell
            is_header_cell = row_idx in table_data.header_rows or col_idx in table_data.header_cols

            # Check for top heading using header information (skip if current cell is a heading)
            if not is_header_cell and col_idx in table_data.col_headers:
                # Get the headers for this column
                col_headers = table_data.col_headers[col_idx]
                if col_headers:
                    # Use the first header as the top heading
                    _, top_heading = col_headers[0]
                    if top_heading and "\n" not in top_heading:
                        test_data["top_heading"] = top_heading

            # Check for left heading using header information (skip if current cell is a heading)
            if not is_header_cell and row_idx in table_data.row_headers:
                # Get the headers for this row
                row_headers = table_data.row_headers[row_idx]
                if row_headers:
                    # Use the first header as the left heading
                    _, left_heading = row_headers[0]
                    if left_heading and "\n" not in left_heading:
                        test_data["left_heading"] = left_heading

            # Only add the test if we have at least one relation
            if len(test_data) > 6:  # 6 is the number of required fields
                # Verify that the test passes with the current table HTML
                # Create the actual test object
                test_obj = TableTest(
                    pdf=test_data["pdf"],
                    page=test_data["page"],
                    id=test_data["id"],
                    type=test_data["type"],
                    cell=test_data["cell"],
                    max_diffs=test_data["max_diffs"],
                    up=test_data.get("up", ""),
                    down=test_data.get("down", ""),
                    left=test_data.get("left", ""),
                    right=test_data.get("right", ""),
                    top_heading=test_data.get("top_heading", ""),
                    left_heading=test_data.get("left_heading", ""),
                )

                # Extract just the relevant table HTML
                tables = soup.find_all("table")
                if table_idx < len(tables):
                    table_html = str(tables[table_idx])

                    # Run the test against the original HTML
                    passed, explanation = test_obj.run(table_html)
                else:
                    # Shouldn't happen, but handle it gracefully
                    passed = False

                # Only add tests that pass
                if passed:
                    tests.append(test_data)

            if len(tests) > 25:
                break

    # Step 3: Generate TextPresenceTests for main body content
    # Make a copy of the soup for the main content
    main_soup = BeautifulSoup(str(soup), "html.parser")

    # Remove headers, footers, and tables from the main_soup
    for element in main_soup.find_all(["header", "footer", "table", "head"]):
        element.extract()

    # Convert superscripts and subscripts in the main soup
    convert_superscripts_subscripts(main_soup)

    full_text = main_soup.get_text().strip()

    sentences = []
    for paragraph in process(full_text):
        for sentence in paragraph:
            # Convert token sequence to string and clean it
            sentence_str = ""
            for token in sentence:
                sentence_str += token.spacing + token.value

            sentence_str = sentence_str.strip()

            if sentence_str:
                sentences.append(sentence_str)

    # Add a few random ordering tests
    all_indexes = list(range(len(sentences)))
    random.shuffle(all_indexes)
    random_pairs = [(all_indexes[i * 2], all_indexes[i * 2 + 1]) for i in range(len(all_indexes) // 2)]
    random_pairs = [(min(i, j), max(i, j)) for (i, j) in random_pairs]

    num_order_tests = 0
    for i, j in random_pairs:
        first_sentence = sentences[i]
        second_sentence = sentences[j]

        if len(first_sentence) < 10 or len(second_sentence) < 10:
            continue

        if "\n" in first_sentence:
            first_sentence = first_sentence.split("\n")[0].strip()
        if "\n" in second_sentence:
            second_sentence = second_sentence.split("\n")[0].strip()

        max_diffs = round(max(len(first_sentence), len(second_sentence)) * 0.05)

        # Too big of a length discrepancy causes issues
        if max_diffs > len(first_sentence) // 2 or max_diffs > len(second_sentence) // 2:
            continue

        tests.append(
            {
                "pdf": pdf_filename,
                "page": page_num,
                "id": f"{pdf_id}_order_{uuid.uuid4().hex[:8]}",
                "type": TestType.ORDER.value,
                "before": first_sentence,
                "after": second_sentence,
                "max_diffs": max_diffs,
            }
        )
        num_order_tests += 1

        if num_order_tests > 5:
            break

    # Final test filtering out stage

    # Now double check that the absent tests don't find any matches in the full_text
    # If they do, filter them out
    tests = [t for t in tests if t["type"] != "absent" or t["text"] not in full_text]

    # Remove any tests where text-based fields have no alphanumeric characters, contain LaTeX, or contain Unicode super/subscripts
    text_fields = ["text", "cell", "before", "after", "up", "down", "left", "right", "top_heading", "left_heading"]

    def contains_alphanumeric(value):
        return any(c.isalnum() for c in value) if isinstance(value, str) else False

    def contains_latex(value):
        if not isinstance(value, str):
            return False
        # Check for LaTeX delimiters
        latex_patterns = [r"\(", r"\)", r"\[", r"\]"]
        return any(pattern in value for pattern in latex_patterns)

    def contains_unicode_super_or_subscripts(value):
        if not isinstance(value, str):
            return False

        # Unicode ranges for superscripts and subscripts
        superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ"
        subscript_chars = "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ"

        return any(c in superscript_chars or c in subscript_chars for c in value)

    filtered_tests = []
    for test in tests:
        # Check all text fields in the test for alphanumeric content, LaTeX, and Unicode super/subscripts
        all_valid = True
        for field in text_fields:
            if field in test:
                # Skip test if field has no alphanumeric characters
                if not contains_alphanumeric(test[field]):
                    all_valid = False
                    break
                # Skip test if field contains LaTeX delimiters
                if contains_latex(test[field]):
                    all_valid = False
                    break
                # Skip test if field contains Unicode super or subscripts
                if contains_unicode_super_or_subscripts(test[field]):
                    all_valid = False
                    break
        if all_valid:
            filtered_tests.append(test)

    tests = filtered_tests

    # Remove duplicate tests (identical on everything but the id field)
    unique_tests = []
    test_signatures = set()

    for test in tests:
        # Create a signature for the test by using all fields except 'id'
        test_dict = test.copy()
        test_dict.pop("id")

        # Convert dict to a sorted tuple of items for hashability
        test_signature = tuple(sorted((k, str(v)) for k, v in test_dict.items()))

        # Only add the test if we haven't seen an identical one
        if test_signature not in test_signatures:
            test_signatures.add(test_signature)
            unique_tests.append(test)

    return unique_tests


def process_pdf(pdf_info, args, client):
    """Process a single PDF, render a random page, and create an HTML template."""
    s3_path, index = pdf_info

    # Create a unique folder for each PDF in the temp directory
    pdf_id = f"pdf_{index:05d}"
    temp_pdf_dir = os.path.join(args.temp_dir, pdf_id)
    os.makedirs(temp_pdf_dir, exist_ok=True)

    # Determine if we should log table test verification
    verbose_table_testing = args.verbose

    # Download PDF to local temp directory
    local_pdf_path = os.path.join(temp_pdf_dir, "document.pdf")
    if not download_s3_pdf(s3_path, local_pdf_path):
        print(f"Failed to download PDF from {s3_path}")
        return None

    try:
        # Get page count using pypdf
        reader = pypdf.PdfReader(local_pdf_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"PDF has no pages: {s3_path}")
            return None

        # Select a random page
        page_num = random.randint(1, num_pages)

        # Render the page as a base64 PNG
        image_base64 = render_pdf_to_base64png(local_pdf_path, page_num, target_longest_image_dim=2048)

        # Generate HTML from the image
        html_content = generate_html_from_image(client, image_base64)
        if not html_content:
            print(f"Failed to generate HTML for {s3_path}, page {page_num}")
            return None

        # Create output directories
        html_dir = os.path.join(args.output_dir, "html")
        pdfs_dir = os.path.join(args.output_dir, "pdfs")
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(pdfs_dir, exist_ok=True)

        # Save HTML to output directory
        html_path = os.path.join(html_dir, f"{pdf_id}_page{page_num}.html")
        with open(html_path, "w") as f:
            f.write(html_content)

        # Extract the page and save as PDF
        original_pdf_path = os.path.join(pdfs_dir, f"{pdf_id}_page{page_num}_original.pdf")
        if not extract_page_from_pdf(local_pdf_path, original_pdf_path, page_num):
            print(f"Failed to extract page {page_num} from {local_pdf_path}")

        # Render PDF using Playwright if not skipped
        playwright_pdf_path = None
        render_success = False
        playwright_pdf_filename = f"{pdf_id}_page{page_num}.pdf"  # This will be used in the tests

        if not args.skip_playwright:
            playwright_pdf_path = os.path.join(pdfs_dir, playwright_pdf_filename)

            try:
                # Get PNG dimensions
                png_width, png_height = get_png_dimensions_from_base64(image_base64)

                # Run the async function in the synchronous context
                render_success = asyncio.run(render_pdf_with_playwright(html_content, playwright_pdf_path, png_width, png_height))

                if render_success:
                    print(f"Successfully rendered with Playwright: {playwright_pdf_path}")
                else:
                    print(f"Failed to render as a single page PDF: {playwright_pdf_path}")
                    playwright_pdf_path = None
            except Exception as e:
                print(f"Failed to render with Playwright: {e}")
                playwright_pdf_path = None
                render_success = False

        # If playwright rendering failed and was required, return None to skip this test
        if not args.skip_playwright and not render_success:
            return None

        # Generate tests from the HTML content
        # Use the playwright rendered PDF path for tests
        tests = generate_tests_from_html(html_content, pdf_id, page_num, verbose_table_testing)

        # Update the PDF path in all tests to use the playwright rendered PDF
        for test in tests:
            test["pdf"] = playwright_pdf_filename

        # Log table test stats if verbose
        if verbose_table_testing:
            table_tests = [t for t in tests if t["type"] == TestType.TABLE.value]
            print(f"Generated {len(table_tests)} table tests for {pdf_id}, page {page_num} (passed verification)")

        return {
            "pdf_id": pdf_id,
            "s3_path": s3_path,
            "page_number": page_num,
            "html_path": html_path,
            "original_pdf_path": original_pdf_path,
            "playwright_pdf_path": playwright_pdf_path,
            "tests": tests,
            "num_tests": len(tests),
        }
    except Exception as e:
        print(f"Error processing {s3_path}: {e}")
        return None
    finally:
        # Clean up temp directory for this PDF
        if os.path.exists(temp_pdf_dir):
            subprocess.run(["rm", "-rf", temp_pdf_dir])


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to HTML templates and render with Playwright")
    parser.add_argument("--input_list", required=True, help="Path to a file containing S3 paths to PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to store extracted pages and tests")
    parser.add_argument("--temp_dir", default="/tmp/mine_tables", help="Directory for temporary files")
    parser.add_argument("--max_tests", type=int, default=100, help="Maximum number of tests to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel threads to use")
    parser.add_argument("--api_key", help="Claude API key (or set ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--skip_playwright", action="store_true", help="Skip Playwright PDF rendering")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output including table test verification")
    args = parser.parse_args()

    # Ensure output and temp directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Get API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api_key or set ANTHROPIC_API_KEY environment variable.")
        return

    # Initialize Claude client
    client = Anthropic(api_key=api_key)

    # Reservoir sampling implementation
    s3_paths = []
    with open(args.input_list, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if not line:
                continue

            if i < 100000:
                s3_paths.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < 100000:
                    s3_paths[j] = line

    print(f"Found {len(s3_paths)} PDF paths in input list")

    # Shuffle and limit to max_tests
    random.shuffle(s3_paths)
    s3_paths = s3_paths[: args.max_tests]

    # Initialize synthetic.json as a JSONL file (empty initially)
    synthetic_json_path = os.path.join(args.output_dir, "synthetic.jsonl")
    open(synthetic_json_path, "w").close()  # Create empty file

    # Counter for test statistics
    test_counter = 0
    test_types = {"present": 0, "absent": 0, "table": 0, "order": 0}
    results = []

    # Initialize a threading lock for file access
    import threading

    file_lock = threading.Lock()

    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        # Submit all tasks
        futures = {executor.submit(process_pdf, (s3_path, i), args, client): s3_path for i, s3_path in enumerate(s3_paths)}

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing PDFs"):
            s3_path = futures[future]
            try:
                result = future.result()
                if result and result.get("tests"):
                    results.append(result)

                    # Append tests to synthetic.json as they're created (JSONL format)
                    with file_lock:
                        # Append each test as a separate JSON line
                        with open(synthetic_json_path, "a") as f:
                            for test in result["tests"]:
                                f.write(json.dumps(test) + "\n")

                        # Update counters
                        test_counter += len(result["tests"])
                        for test in result["tests"]:
                            test_type = test.get("type", "")
                            if test_type in test_types:
                                test_types[test_type] += 1

                        print(f"Added {len(result['tests'])} tests from {result['pdf_id']}, total: {test_counter}")
            except Exception as e:
                print(f"Error processing {s3_path}: {e}")

    print(f"Generated {len(results)} HTML templates")

    # Print summary of Playwright rendering results
    playwright_success = sum(1 for r in results if r and r.get("playwright_pdf_path"))
    if not args.skip_playwright:
        print(f"Playwright PDF rendering: {playwright_success}/{len(results)} successful")

    print(f"Saved {test_counter} tests to {synthetic_json_path}")

    # Print summary of generated tests
    print(f"Generated a total of {test_counter} tests across {len(results)} templates")

    # Print test type distribution
    if test_counter > 0:
        print("Test type distribution:")
        for test_type, count in test_types.items():
            print(f"  - {test_type}: {count} tests")


if __name__ == "__main__":
    main()
