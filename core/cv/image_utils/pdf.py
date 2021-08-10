from PIL import Image
import fitz  # fitz: pip install PyMuPDF


def pdf2images(doc, zoom=2, color='RGB'):
    """pdf to images
    example: 
        doc = fitz.open(/path/to/pdf)
        images = pdf2images(doc)
    example: 
        stream = open(/path/to/pdf, 'rb')
        doc = fitz.open(stream)
        images = pdf2images(doc)
    example: 
        doc = fitz.open(stream=bytes, filetype='bytes')
        images = pdf2images(doc)
    """
    mat = fitz.Matrix(zoom, zoom)
    images = []
    # mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
    # for pg in range(doc.pageCount):
        # page = doc[pg]
    for page in doc:
        pix = page.getPixmap(matrix=mat, alpha=False)
        images.append(Image.frombytes(color, [pix.width, pix.height], pix.samples))

    return images

if __name__ == "__main__":
    import sys
    doc = fitz.open(sys.argv[1])
    images = pdf2images(doc)
    print(len(images), images[0].size)