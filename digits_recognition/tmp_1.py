import io

from bottle import run, get, HTTPResponse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig = Figure(figsize=[6,6])
ax = fig.add_axes([.1,.1,.8,.8])
ax.scatter([0.2, 0.3], [0.25, 0.35])
canvas = FigureCanvasAgg(fig)

buf = io.BytesIO()
canvas.print_png(buf)
data = buf.getvalue()

headers = {
    'Content-Type': 'image/png',
    'Content-Length': len(data)
}

@get('/')
def hello():
    print(data)
    return HTTPResponse(body=data, headers=headers)
    # return HTTPResponse(body=data)


run(port=8080)