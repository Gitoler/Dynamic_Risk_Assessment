"""
This class is to add content into the pdf
author: Ung Van Tuan
date: July 10th 2023
"""
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        """Function for the page header"""
        # Arial bold 16
        self.set_font('Arial', 'B', 16)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(40, 10, 'Model Report', 1, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        """Function for the page footer"""
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def chapter_title(self, num, label):
        """Function for the chapter title"""
        # Arial 12
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, 'Chapter %d : %s' % (num, label), 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_subtitle(self, subtile):
        """Funtion for the chapter subtitle"""
        self.set_font('Times', '', 14)
        self.cell(0, 10, subtile, 0, 1)
        self.ln(2)

    def chapter_body(self, name):
        """Function for the body content from text file
        Input:
        name: str
        """
        # Read text file
        with open(name, 'rb') as fh:
            txt = fh.read().decode('latin-1')
        # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 5, txt)
        # Line break
        self.ln()

    def print_chapter(self, num, title, name, subtile):
        """Function for printing into the chaper
        Input:
        num: int
        title: str
        name: str
        subtitle: str
        """
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_subtitle(subtile)
        self.chapter_body(name)

    def add_text(self, text):
        """
        Funtion to add text in the chapter body
        Input:
        text: str
        """
        self.set_font('Times', '', 12)
        self.cell(0, 5, text)
        self.ln()

    def image_(self, image_path, *args):
        """Function to add image
        Input:
        image_path: str
        """
        self.image(image_path, args[0], args[1], args[2])
        self.ln()