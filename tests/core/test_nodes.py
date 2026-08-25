from llama_launcher.core.nodes import Node, LOCAL_NODE, connection_for, host_of, valid_ssh_target


def test_local_node_injects_no_connection():
    assert LOCAL_NODE.kind == "local"
    assert connection_for(LOCAL_NODE) == ""
    assert host_of(LOCAL_NODE) == "127.0.0.1"


def test_remote_node_connection_and_host():
    n = Node(name="box-b", kind="remote", connection="box-b",
             ssh_target="me@192.168.1.11:22", binary="podman")
    assert connection_for(n) == "box-b"
    assert host_of(n) == "192.168.1.11"


def test_valid_ssh_target_accepts_user_at_host():
    assert valid_ssh_target("me@10.0.0.2") is True


def test_valid_ssh_target_accepts_plain_host():
    assert valid_ssh_target("host") is True


def test_valid_ssh_target_accepts_host_with_port():
    assert valid_ssh_target("user@host:22") is True


def test_valid_ssh_target_rejects_empty():
    assert valid_ssh_target("") is False


def test_valid_ssh_target_rejects_leading_dash_proxycommand():
    assert valid_ssh_target("-oProxyCommand=x") is False


def test_valid_ssh_target_rejects_leading_dash_option():
    assert valid_ssh_target("-x") is False


def test_valid_ssh_target_rejects_embedded_space():
    assert valid_ssh_target("a b") is False


def test_valid_ssh_target_rejects_command_injection():
    assert valid_ssh_target("user@host;rm -rf") is False


def test_valid_ssh_target_accepts_bracketed_ipv6():
    assert valid_ssh_target("[2001:db8::1]") is True
    assert valid_ssh_target("me@[2001:db8::1]:22") is True


def test_valid_ssh_target_still_rejects_option_smuggle():
    assert valid_ssh_target("-oProxyCommand=x") is False
    assert valid_ssh_target("") is False


def test_host_of_bracketed_ipv6_without_port():
    # rsplit(':',1) mangled a bracketed IPv6 with no explicit port into a broken
    # host; the bare address must come back so url_host can re-bracket it.
    n = Node(name="v6", kind="remote", connection="v6", ssh_target="[2001:db8::1]")
    assert host_of(n) == "2001:db8::1"


def test_host_of_bracketed_ipv6_with_port_and_user():
    n = Node(name="v6", kind="remote", connection="v6", ssh_target="me@[2001:db8::1]:22")
    assert host_of(n) == "2001:db8::1"


def test_host_of_bracketed_loopback_ipv6():
    n = Node(name="v6", kind="remote", connection="v6", ssh_target="[::1]")
    assert host_of(n) == "::1"
