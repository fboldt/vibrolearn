class MockHeader():
    def __init__(self, ContentLength=1024):
        self.headers = {'Content-Length': ContentLength}
    def get(self, key, _):
        return self.headers[key]
    
class MockHeadResponse():
    def __init__(self, status_code=200, headers=MockHeader(1024)):
        self.status_code = status_code
        self.headers = headers
