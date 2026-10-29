"""Real loopback HTTP tests for observer isolation, reconnects and truthful state."""

import base64
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import sys
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import isaac_observe
from isaac_observe import ObservationPublisher, ObservationServer, atomic_json, read_json


# Self-authored solid-color JPEG fixtures encoded by FFmpeg, not downloaded assets.
RED = base64.b64decode('/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYxLjE5LjEwMAD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABNAAEBAAAAAAAAAAAAAAAAAAAABgEBAQEAAAAAAAAAAAAAAAAAAAYHEAEAAAAAAAAAAAAAAAAAAAAAEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGAAgAwEiAAIRAAMRAP/aAAwDAQACEQMRAD8AiwEo38AAAAAB/9k=')
BLUE = base64.b64decode('/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYxLjE5LjEwMAD/2wBDAAgEBAQEBAUFBQUFBQYGBgYGBgYGBgYGBgYHBwcICAgHBwcGBgcHCAgICAkJCQgICAgJCQoKCgwMCwsODg4RERT/xABNAAEBAAAAAAAAAAAAAAAAAAAABwEBAQEAAAAAAAAAAAAAAAAAAAUHEAEAAAAAAAAAAAAAAAAAAAAAEQEAAAAAAAAAAAAAAAAAAAAA/8AAEQgAGAAgAwEiAAIRAAMRAP/aAAwDAQACEQMRAD8AjgDf0sAAAAAB/9k=')


def read_part(response):
    if response.readline().strip() != b"--frame":
        raise AssertionError("Missing MJPEG boundary")
    headers = {}
    while line := response.readline().strip():
        key, value = line.decode().split(":", 1)
        headers[key] = value.strip()
    jpeg = response.read(int(headers["Content-Length"]))
    if response.read(2) != b"\r\n":
        raise AssertionError("Incomplete MJPEG image")
    return int(headers["X-Frame-ID"]), jpeg


class SocketPairServer:
    """Replace only the forbidden TCP bind; run the real HTTP handler over sockets."""

    def __init__(self, address, handler):
        self.server_address = address
        self.server_port = address[1]
        self.RequestHandlerClass = handler

    def serve_forever(self):
        pass

    def shutdown(self):
        pass

    def server_close(self):
        pass


class PairConnection:
    def __init__(self, server):
        self.client, self.remote = socket.socketpair()
        self.client.settimeout(3)
        self.server = server
        self.response = None

    def request(self, method, path):
        def handle():
            with self.remote:
                self.server.http.RequestHandlerClass(self.remote, ("127.0.0.1", 0), self.server.http)
        self.thread = threading.Thread(target=handle, daemon=True)
        self.thread.start()
        self.client.sendall(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".encode())

    def getresponse(self):
        self.response = http.client.HTTPResponse(self.client)
        self.response.begin()
        return self.response

    def close(self):
        if self.response is not None:
            self.response.close()
        self.client.close()
        self.thread.join(timeout=1.5)


@contextmanager
def http_server(directory):
    with mock.patch.object(isaac_observe, "ThreadingHTTPServer", SocketPairServer):
        with ObservationServer(directory, port=0) as server:
            yield server


class ObservationTests(unittest.TestCase):
    def test_closed_initialization_streams_release_their_slots(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with http_server(directory) as server:
                for _ in range(6):
                    connection = PairConnection(server)
                    connection.request("GET", "/stream.mjpg")
                    self.assertEqual(connection.getresponse().status, 200)
                    connection.close()
                publisher = ObservationPublisher(directory)
                publisher.publish_jpeg(RED, step=0, sim_time_seconds=0, resets=0)
                connection = PairConnection(server)
                connection.request("GET", "/stream.mjpg")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(read_part(response), (1, RED))
                connection.close()

    def test_http_stream_changes_and_reconnect_uses_latest_frame(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            atomic_json(directory / "summary.json", {"status": "running", "current_stage": "g1"})
            publisher = ObservationPublisher(directory, source="replay")
            with http_server(directory) as server:
                self.assertEqual(server.http.server_address[0], "127.0.0.1")
                publisher.publish_jpeg(RED, step=2, sim_time_seconds=0.04, resets=0, force=True)
                connection = PairConnection(server)
                connection.request("GET", "/stream.mjpg")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(read_part(response), (1, RED))
                publisher.publish_jpeg(BLUE, step=4, sim_time_seconds=0.08, resets=1, force=True)
                self.assertEqual(read_part(response), (2, BLUE))
                response.close()
                connection.close()
                # No viewer is connected; the producer still advances independently.
                publisher.publish_jpeg(RED, step=6, sim_time_seconds=0.12, resets=1, force=True)
                self.assertEqual(read_json(directory / "observation.json")["step"], 6)
                second = PairConnection(server)
                second.request("GET", "/stream.mjpg")
                resumed = second.getresponse()
                self.assertEqual(read_part(resumed), (3, RED))
                resumed.close()
                second.close()
                self.assertGreaterEqual(server.connections_seen, 2)
                self.assertFalse(publisher.disabled)

    def test_routes_do_not_serve_project_files_or_expose_jpeg_in_status(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "private.txt").write_text("not an observation artifact")
            publisher = ObservationPublisher(directory)
            publisher.publish_jpeg(RED, step=2, sim_time_seconds=0.04, resets=0)
            with http_server(directory) as server:
                for path in ("/private.txt", "/../private.txt", "/%2e%2e/private.txt", "/.git/config"):
                    connection = PairConnection(server)
                    connection.request("GET", path)
                    response = connection.getresponse()
                    self.assertEqual(response.status, 404)
                    response.read()
                    connection.close()
                connection = PairConnection(server)
                connection.request("GET", "/status.json")
                response = connection.getresponse()
                state = json.loads(response.read())
                self.assertNotIn("jpeg_base64", state["frame"])
                self.assertEqual(state["frame"]["step"], 2)
                connection.close()
                connection = PairConnection(server)
                connection.request("GET", "/")
                page = connection.getresponse().read().decode()
                self.assertIn("does not stop the run or release the GPU", page)
                connection.close()

    def test_stale_frame_and_completed_run_are_different_states(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            atomic_json(directory / "summary.json", {"status": "running", "current_stage": "gpu"})
            with http_server(directory) as server:
                self.assertEqual(server.state()["view_status"], "checking_gpu")
                atomic_json(directory / "summary.json", {"status": "running", "current_stage": "g1"})
                self.assertEqual(server.state()["view_status"], "initializing")
                publisher = ObservationPublisher(directory)
                publisher.publish_jpeg(RED, step=2, sim_time_seconds=0.04, resets=0)
                snapshot = read_json(directory / "observation.json")
                snapshot["captured_unix_seconds"] = time.time() - 10
                atomic_json(directory / "observation.json", snapshot)
                self.assertEqual(server.state()["view_status"], "stale")
                publisher.phase("encoding")
                self.assertEqual(server.state()["view_status"], "encoding")
                atomic_json(directory / "summary.json", {"status": "passed_g1_recording"})
                self.assertEqual(server.state()["view_status"], "finished")
                self.assertEqual(server.state()["run_status"], "passed_g1_recording")

    def test_invalid_preview_disables_preview_without_changing_run_status(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            atomic_json(directory / "summary.json", {"status": "running"})
            publisher = ObservationPublisher(directory)
            self.assertFalse(publisher.publish_jpeg(b"not a jpeg", step=0, sim_time_seconds=0, resets=0))
            self.assertTrue(publisher.disabled)
            self.assertEqual(read_json(directory / "summary.json")["status"], "running")
            self.assertTrue((directory / "observation-error.log").is_file())

    def test_actual_loopback_tcp_listener_when_permitted(self):
        with tempfile.TemporaryDirectory() as name:
            try:
                server = ObservationServer(Path(name), port=0)
            except PermissionError:
                self.skipTest("Agent sandbox forbids TCP bind; socket-pair HTTP tests still run")
            with server:
                connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=3)
                connection.request("GET", "/status.json")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["view_status"], "initializing")
                connection.close()


if __name__ == "__main__":
    unittest.main()
