from gevent.pywsgi import WSGIServer
from A股强度_dash import server

http_server = WSGIServer(('0.0.0.0', 12000), server)
http_server.serve_forever()