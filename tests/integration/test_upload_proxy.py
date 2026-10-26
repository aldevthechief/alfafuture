"""Opt-in check through a running frontend proxy; invalid fixtures never start PDF processing.

UPLOAD_PROXY_TEST_URL=http://127.0.0.1:5173 python -m pytest tests/integration/test_upload_proxy.py
"""
import http.client
import json
import os
from urllib.parse import urlsplit

import pytest

PROXY=os.environ.get('UPLOAD_PROXY_TEST_URL')


@pytest.mark.skipif(not PROXY,reason='Set UPLOAD_PROXY_TEST_URL to test a running frontend proxy')
@pytest.mark.parametrize('size,status,detail',[
    (round(7.1*1024*1024),422,'Invalid PDF'),
    (50*1024*1024,422,'Invalid PDF'),
    (50*1024*1024+1,413,'PDF exceeds 50 MiB'),
])
def test_upload_limit_reaches_pdf_validation_through_nginx(size,status,detail):
    url=urlsplit(PROXY)
    boundary='upload-proxy-regression'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="proxy-limit-probe.pdf"\r\n'
          'Content-Type: application/pdf\r\n\r\n').encode()+b'x'*size+f'\r\n--{boundary}--\r\n'.encode()
    connection_type=http.client.HTTPSConnection if url.scheme=='https' else http.client.HTTPConnection
    connection=connection_type(url.hostname,url.port,timeout=60)
    try:
        connection.request('POST',url.path.rstrip('/')+'/v1/reports/upload',body,
                           {'Content-Type':'multipart/form-data; boundary='+boundary,'X-User-ID':'upload-proxy-test'})
        response=connection.getresponse()
        payload=response.read()
        assert response.status==status,payload[:400]
        assert 'application/json' in response.getheader('Content-Type',''),payload[:400]
        assert json.loads(payload)['detail']==detail
    finally:
        connection.close()
