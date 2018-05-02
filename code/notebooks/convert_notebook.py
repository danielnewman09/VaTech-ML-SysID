import sys
from nbconvert import HTMLExporter
import nbformat


print(sys.argv)


notebook_file = 'Create-Model.ipynb'
html_exporter = HTMLExporter()
nb = nbformat.reads(open(notebook_file, 'r').read(), as_version=4)

(body, resources) = html_exporter.from_notebook_node(nb)
html_file = notebook_file.replace(".ipynb", ".html")
html_file_writer = open(html_file, 'w')
html_file_writer.write(body)
html_file_writer.close()

read_navbar = open("navbar.txt", 'r').read()
read_disqus = open("disqus.txt", 'r').read()
read_css = open("main_css.txt", 'r').read()
read_ga = open("google_analytics.txt","r").read()

if read_navbar not in body:
    body = body.replace("<body>", "<body>\n" + read_navbar)

if read_ga not in body:
    body = body.replace("</body", read_ga + "\n</body>")

if read_disqus not in body:
    body = body.replace("</body", read_disqus + "\n</body>")

if read_css not in body:
    body = body.replace("</title>", "</title>\n" + read_css)