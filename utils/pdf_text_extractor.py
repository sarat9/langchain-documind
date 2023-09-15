from PyPDF2 import PdfReader

def extract_text_from_pdf(path_to_pdf_file):
    pdf_text = ""
    with open(path_to_pdf_file, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PdfReader(file)

        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # print("Starting Loop for total pages : " + num_pages)

        # Loop through each page and extract the text content
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]

            # Extract the Text from PDF
            page_text = page.extract_text()

            # Do something with the extracted text
            print(f"Page {page_num + 1}: Loading...")
            # print(page_text)

            pdf_text += page_text

    return pdf_text