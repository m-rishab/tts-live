# gunicorn.conf.py
bind = "127.0.0.1:5000"
workers = 1
worker_class = "eventlet"
loglevel = "debug"

# SSL settings (assuming these were intended to be included)
keyfile = None
certfile = None
ssl_version = 2
cert_reqs = 0
ca_certs = None
suppress_ragged_eofs = True
do_handshake_on_connect = False
ciphers = None

# Additional settings
proxy_protocol = False
proxy_allow_ips = '127.0.0.1'
raw_paste_global_conf = []
strip_header_spaces = False
permit_unconventional_http_method = False
permit_unconventional_http_version = False
casefold_http_method = False
header_map = "drop"
tolerate_dangerous_framing = False
