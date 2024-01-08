#
# A simple HTTP server that generates content based on data received from
# client requests.
#
# Created: 8 January 2024
#

import re

from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi

import Dictionary_SvdEngine

hostName = '0.0.0.0'
serverPort = 8080

svd_engine = None
sdxl_engine = None

class Dictionary_SvdServer(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.startswith('/vid/'):
            # Retrieve the previously generated video
            self.serve_video()

    def do_POST(self):
        # Doesn't do anything with posted data
        # TODO replace cgi with non-deprecated equivalents
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        else:
            postvars = {}
        # print(postvars)
        if 'image' in postvars:
            image_bytes = postvars['image'][0]
            video_bytes = self.generate_video(image_bytes)
            if video_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", "inline")
                self.send_header("Content-Length", len(video_bytes))
                self.end_headers()
                self.wfile.write(video_bytes)
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes("<html><body><h1>POST!</h1></body></html>", "utf-8"))

    def generate_video(self, image):
        print(f"Generating video for image")
        return svd_engine.generate_video(image)

    def serve_video(self):
        video_id = re.match('^/vid/(.+)', self.path).group(1)
        print(f"Serving video with id '{video_id}'")


if __name__ == "__main__":

    # Create the SVD engine
    svd_engine = Dictionary_SvdEngine.Dictionary_SvdEngine()

    # Create and start the web server
    webServer = HTTPServer((hostName, serverPort), Dictionary_SvdServer)
    print(f"Server started http://{hostName}:{serverPort}")
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    # Shutdown the webserver when we're done with it
    webServer.server_close()
    print("Server stopped.")
