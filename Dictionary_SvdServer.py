#
# A simple HTTP server that generates content based on data received from
# client requests.
#
# Created: 8 January 2024
#

import os
import re

from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
from urllib.parse import urlparse, parse_qs

import Dictionary_SdxlEngine
import Dictionary_SvdEngine

hostName = '0.0.0.0'
serverPort = 8080

svd_engine = None
sdxl_engine = None

class Dictionary_SvdServer(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.startswith('/img/'):
            # Generate image based on the prompt
            query_components = parse_qs(urlparse(self.path).query)
            prompt = query_components["prompt"][0]
            seed = None
            steps = 40
            if "seed" in query_components:
                seed = int(query_components["seed"][0])
            if "steps" in query_components:
                steps = int(query_components["steps"][0])
            image_bytes = self.generate_image(prompt, seed = seed, steps = steps)
            if image_bytes:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Disposition", "inline")
                self.send_header("Content-Length", len(image_bytes))
                self.end_headers()
                self.wfile.write(image_bytes)
        elif self.path.startswith('/vid/'):
            # Retrieve the previously generated video
            self.serve_video()

    def do_POST(self):
        # Get the query parameters
        request_url = urlparse(self.path)
        query_components = parse_qs(request_url.query)
        mode = 'data'
        if 'mode' in query_components:
            mode = query_components['mode'][0]
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
            (video_path, video_bytes) = self.generate_video(image_bytes)
            if mode == 'data':
                if video_bytes:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Disposition", "inline")
                    self.send_header("Content-Length", len(video_bytes))
                    self.end_headers()
                    self.wfile.write(video_bytes)
            elif mode == 'url':
                if video_path:
                    # Get just the video id, not the rest of its path
                    video_id = os.path.splitext(os.path.basename(video_path.replace('\\','/'))
                        )[0]
                    # Build the URL the client should use to access the video
                    video_url = request_url._replace(
                        path = f"/vid/{video_id}",
                        query = None).geturl()
                    if video_url.find('http') == -1:
                        # No host in the url, use the one from the request header
                        video_url = "http://" + self.headers['Host'] + video_url
                    # Write the response
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", len(video_url))
                    self.end_headers()
                    self.wfile.write(video_url.encode(encoding = 'utf-8'))
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes("<html><body><h1>POST!</h1></body></html>", "utf-8"))

    def generate_image(self, prompt, seed = None, steps = 40):
        print('Generating video for prompt: {}'.format(prompt))
        return sdxl_engine.generate_image(prompt, seed = seed, steps = steps)

    def generate_video(self, image):
        print(f"Generating video for image")
        return svd_engine.generate_video(image)

    def serve_video(self):
        video_id = re.match('^/vid/(.+)', self.path).group(1)
        print(f"Serving video with id '{video_id}'")
        video_bytes = svd_engine.get_video(video_id)
        if video_bytes:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", len(video_bytes))
            self.end_headers()
            self.wfile.write(video_bytes)


if __name__ == "__main__":

    # Create the SVD engine
    svd_engine = Dictionary_SvdEngine.Dictionary_SvdEngine()

    # Create the SDXL engine
    sdxl_engine = Dictionary_SdxlEngine.Dictionary_SdxlEngine()

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
