"""Tests for scripts/vast_box.py — pure parts only, no live Vast API.

Covers offer parsing (control chars → strict=False), eligibility/ranking,
the dud-detection state machine, ~/.ssh/config rewriting, the ownership
ledger, endpoint extraction, the provisioning plan, and the fake-HTTP-driven
fetch/rent paths.
"""

from __future__ import annotations

import json

import pytest

from core.result import Err, Ok
from scripts.vast_box import (
    DEFAULT_QUERY,
    TEMPLATE_ID,
    Endpoint,
    Offer,
    WaitState,
    advance,
    authorized_key_line,
    build_query,
    bundles_url,
    eligible_offers,
    fetch_offers,
    label_conflict,
    ledger_add,
    ledger_owns,
    ledger_remove,
    parse_instance,
    parse_instances,
    parse_offers,
    provision_plan,
    race_step,
    rewrite_ssh_config,
    ssh_config_block,
)


def _offer(
    offer_id: int, dph: float = 0.40, machine: int = 1, direct: int = 4
) -> Offer:
    return Offer(
        offer_id=offer_id,
        machine_id=machine,
        gpu_name="RTX 4090",
        num_gpus=1,
        dph_total=dph,
        reliability=0.995,
        inet_down=800.0,
        inet_up=700.0,
        direct_ports=direct,
        disk_space=100.0,
        geolocation="SE",
    )


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------


class TestBuildQuery:
    def test_defaults_match_validated_filter(self) -> None:
        q = build_query()
        assert q == DEFAULT_QUERY
        assert q["gpu_name"] == {"eq": "RTX 4090"}
        assert q["direct_port_count"] == {"gt": 0}
        assert q["order"] == [["dph_total", "asc"]]
        assert q["type"] == "ask"

    def test_gpu_and_dph_overrides(self) -> None:
        q = build_query(gpu_name="RTX 5090", max_dph=0.7)
        assert q["gpu_name"] == {"eq": "RTX 5090"}
        assert q["dph_total"] == {"lt": 0.7}
        # override must not mutate the shared default
        assert DEFAULT_QUERY["gpu_name"] == {"eq": "RTX 4090"}
        assert "dph_total" not in DEFAULT_QUERY

    def test_bundles_url_is_get_with_urlencoded_json(self) -> None:
        url = bundles_url(build_query())
        assert url.startswith("https://console.vast.ai/api/v0/bundles/?q=")
        assert "%22gpu_name%22" in url  # JSON went through quote()
        assert " " not in url


# ---------------------------------------------------------------------------
# Offer parsing + eligibility
# ---------------------------------------------------------------------------


class TestParseOffers:
    def test_parses_payload_with_raw_control_chars(self) -> None:
        # Vast payloads contain literal control characters inside strings;
        # strict json.loads raises, strict=False must not.
        text = (
            '{"offers": [{"id": 111, "machine_id": 9, "gpu_name": "RTX 4090",'
            ' "num_gpus": 1, "dph_total": 0.39, "reliability2": 0.997,'
            ' "inet_down": 850.1, "inet_up": 610.0, "direct_port_count": 8,'
            ' "disk_space": 200.5, "geolocation": "Sw\x07eden"}]}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)  # proves the fixture actually exercises strict=False
        match parse_offers(text):
            case Ok(offers):
                assert len(offers) == 1
                assert offers[0].offer_id == 111
                assert offers[0].direct_ports == 8
                assert "eden" in offers[0].geolocation
            case Err(msg):
                pytest.fail(msg)

    def test_missing_fields_default_not_raise(self) -> None:
        match parse_offers('{"offers": [{"id": 5}]}'):
            case Ok(offers):
                assert offers[0].dph_total == 0.0
                assert offers[0].direct_ports == 0
            case Err(msg):
                pytest.fail(msg)

    def test_non_json_and_missing_array_are_err(self) -> None:
        assert isinstance(parse_offers("<html>502</html>"), Err)
        assert isinstance(parse_offers('{"error": "nope"}'), Err)


class TestEligibleOffers:
    def test_drops_zero_direct_ports(self) -> None:
        offers = (_offer(1, direct=0), _offer(2, direct=4))
        assert [o.offer_id for o in eligible_offers(offers)] == [2]

    def test_excludes_dud_machines(self) -> None:
        offers = (_offer(1, machine=7), _offer(2, machine=8))
        got = eligible_offers(offers, exclude_machines=frozenset({7}))
        assert [o.offer_id for o in got] == [2]

    def test_sorted_cheapest_first(self) -> None:
        offers = (_offer(1, dph=0.55), _offer(2, dph=0.30), _offer(3, dph=0.41))
        assert [o.offer_id for o in eligible_offers(offers)] == [2, 3, 1]


# ---------------------------------------------------------------------------
# Instance parsing — direct AND proxy endpoints
# ---------------------------------------------------------------------------


class TestParseInstance:
    def test_direct_from_ports_and_public_ip(self) -> None:
        inst = parse_instance(
            {
                "id": 42,
                "machine_id": 900,
                "label": "mert",
                "actual_status": "running",
                "public_ipaddr": "1.2.3.4",
                "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]},
                "ssh_host": "ssh7.vast.ai",
                "ssh_port": 25722,
            }
        )
        assert inst.direct == Endpoint("1.2.3.4", 41234)
        assert inst.proxy == Endpoint("ssh7.vast.ai", 25722)

    def test_null_ports_yields_proxy_only(self) -> None:
        # `ports: null` on a healthy box is real (cost ~9 wasted boxes once);
        # it must parse to direct=None, NOT be treated as failure.
        inst = parse_instance(
            {
                "id": 43,
                "actual_status": "loading",
                "ports": None,
                "ssh_host": "ssh5.vast.ai",
                "ssh_port": 11,
            }
        )
        assert inst.direct is None
        assert inst.proxy == Endpoint("ssh5.vast.ai", 11)

    def test_parse_instances_array(self) -> None:
        match parse_instances('{"instances": [{"id": 1}, {"id": 2}]}'):
            case Ok(instances):
                assert [i.instance_id for i in instances] == [1, 2]
            case Err(msg):
                pytest.fail(msg)

    def test_label_conflict(self) -> None:
        instances = (parse_instance({"id": 1, "label": "mert"}),)
        assert label_conflict(instances, "mert") is not None
        assert label_conflict(instances, "other") is None


# ---------------------------------------------------------------------------
# Dud-detection state machine
# ---------------------------------------------------------------------------


class TestAdvance:
    def test_booting_waits(self) -> None:
        s = WaitState(started=0.0, running_since=None)
        s, v = advance(s, status="loading", port_listening=False, now=100.0)
        assert v == "wait"
        assert s.running_since is None

    def test_running_starts_dud_clock(self) -> None:
        s = WaitState(started=0.0, running_since=None)
        s, v = advance(s, status="running", port_listening=False, now=300.0)
        assert v == "wait"
        assert s.running_since == 300.0

    def test_running_and_listening_is_ready(self) -> None:
        s = WaitState(started=0.0, running_since=300.0)
        _, v = advance(s, status="running", port_listening=True, now=310.0)
        assert v == "ready"

    def test_dud_after_10_minutes_running_without_ssh(self) -> None:
        # The rule that bit us: running box whose port 22 never listens.
        s = WaitState(started=0.0, running_since=300.0)
        _, v = advance(s, status="running", port_listening=False, now=899.9)
        assert v == "wait"
        _, v = advance(s, status="running", port_listening=False, now=900.0)
        assert v == "dud"

    def test_dud_clock_keyed_to_running_not_start(self) -> None:
        # 700s of booting then running: dud clock restarts at running.
        s = WaitState(started=0.0, running_since=None)
        s, v = advance(s, status="running", port_listening=False, now=700.0)
        assert v == "wait"
        _, v = advance(s, status="running", port_listening=False, now=1250.0)
        assert v == "wait"  # only 550s running
        _, v = advance(s, status="running", port_listening=False, now=1300.0)
        assert v == "dud"

    def test_never_running_boot_timeout(self) -> None:
        s = WaitState(started=0.0, running_since=None)
        _, v = advance(s, status="loading", port_listening=False, now=1800.0)
        assert v == "boot_timeout"

    def test_listening_probe_ignored_unless_running(self) -> None:
        s = WaitState(started=0.0, running_since=None)
        _, v = advance(s, status="loading", port_listening=True, now=10.0)
        assert v == "wait"


# ---------------------------------------------------------------------------
# Race pool (parallel dud-avoidance)
# ---------------------------------------------------------------------------


class TestRaceStep:
    def test_first_ready_is_a_winner_the_rest_wait(self) -> None:
        states = {1: WaitState(0.0, 300.0), 2: WaitState(0.0, None)}
        obs = {1: ("running", True), 2: ("loading", False)}
        _, verdicts = race_step(states, obs, now=310.0)
        assert verdicts[1] == "ready"
        assert verdicts[2] == "wait"

    def test_dud_detected_per_box(self) -> None:
        states = {1: WaitState(0.0, 300.0)}
        obs = {1: ("running", False)}  # running but port never opens
        _, verdicts = race_step(states, obs, now=901.0, dud_after=600.0)
        assert verdicts[1] == "dud"

    def test_boot_timeout_per_box(self) -> None:
        states = {7: WaitState(0.0, None)}
        obs = {7: ("loading", False)}
        _, verdicts = race_step(states, obs, now=1801.0, boot_timeout=1800.0)
        assert verdicts[7] == "boot_timeout"

    def test_boxes_advance_independently(self) -> None:
        states = {1: WaitState(0.0, None), 2: WaitState(0.0, None)}
        obs = {1: ("running", False), 2: ("loading", False)}
        new_states, _ = race_step(states, obs, now=100.0)
        assert new_states[1].running_since == 100.0  # box 1 started its dud clock
        assert new_states[2].running_since is None  # box 2 still booting


class TestQuarantine:
    def test_roundtrip_and_merge(self, tmp_path, monkeypatch) -> None:
        import scripts.vast_box as vb

        monkeypatch.setattr(vb, "QUARANTINE_PATH", tmp_path / "q")
        assert vb.load_quarantine() == frozenset()
        vb.add_quarantine({111, 222})
        assert vb.load_quarantine() == frozenset({111, 222})
        vb.add_quarantine({222, 333})  # union, no dupes
        assert vb.load_quarantine() == frozenset({111, 222, 333})

    def test_ignores_nondigit_lines(self, tmp_path, monkeypatch) -> None:
        import scripts.vast_box as vb

        (tmp_path / "q").write_text("111\n# a comment\n\n222\n")
        monkeypatch.setattr(vb, "QUARANTINE_PATH", tmp_path / "q")
        assert vb.load_quarantine() == frozenset({111, 222})

    def test_empty_set_is_noop(self, tmp_path, monkeypatch) -> None:
        import scripts.vast_box as vb

        monkeypatch.setattr(vb, "QUARANTINE_PATH", tmp_path / "q")
        vb.add_quarantine(set())
        assert not (tmp_path / "q").exists()


# ---------------------------------------------------------------------------
# ~/.ssh/config rewrite
# ---------------------------------------------------------------------------


class TestRewriteSshConfig:
    def test_appends_when_absent(self) -> None:
        base = "Host pi-storage\n    HostName pi.example\n"
        match rewrite_ssh_config(base, "vast", "1.2.3.4", 41234):
            case Ok(text):
                assert "Host pi-storage" in text
                assert "Host vast" in text
                assert "HostName 1.2.3.4" in text
                assert "Port 41234" in text
                assert "User root" in text
                assert "StrictHostKeyChecking accept-new" in text
                assert "UserKnownHostsFile /dev/null" in text
            case Err(msg):
                pytest.fail(msg)

    def test_replaces_existing_block_preserving_neighbours(self) -> None:
        base = (
            "Host pi-storage\n    HostName pi.example\n\n"
            "Host vast\n    HostName 9.9.9.9\n    Port 999\n    User olduser\n\n"
            "Host github.com\n    User git\n"
        )
        match rewrite_ssh_config(base, "vast", "1.2.3.4", 41234):
            case Ok(text):
                assert "9.9.9.9" not in text
                assert "olduser" not in text
                assert text.count("Host vast") == 1
                assert "HostName pi.example" in text
                assert "Host github.com" in text
                assert "HostName 1.2.3.4" in text
            case Err(msg):
                pytest.fail(msg)

    def test_refuses_multi_alias_block(self) -> None:
        base = "Host vast vast-old\n    HostName 9.9.9.9\n"
        assert isinstance(rewrite_ssh_config(base, "vast", "1.2.3.4", 1), Err)

    def test_empty_config(self) -> None:
        match rewrite_ssh_config("", "vast", "1.2.3.4", 41234):
            case Ok(text):
                assert text.startswith("Host vast")
                assert text.endswith("\n")
            case Err(msg):
                pytest.fail(msg)

    def test_block_shape(self) -> None:
        block = ssh_config_block("vast", "1.2.3.4", 5)
        assert block[0] == "Host vast"
        assert all(line.startswith("    ") for line in block[1:])


# ---------------------------------------------------------------------------
# Ownership ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_add_owns_remove_roundtrip(self) -> None:
        led = ledger_add({}, 40765622, "taste-embed", 60, 1000.0)
        assert ledger_owns(led, 40765622)
        assert led["40765622"]["label"] == "taste-embed"
        led2 = ledger_remove(led, 40765622)
        assert not ledger_owns(led2, 40765622)
        # pure: original untouched
        assert ledger_owns(led, 40765622)

    def test_does_not_own_foreign_box(self) -> None:
        assert not ledger_owns({}, 12345)


# ---------------------------------------------------------------------------
# Provision plan + pubkey line
# ---------------------------------------------------------------------------


class TestProvisionPlan:
    def test_clone_is_github_shallow_with_branch(self) -> None:
        cmds = [cmd for _, cmd in provision_plan("my-branch", roformer=False)]
        clone = cmds[0]
        assert "git clone --depth 1 --branch my-branch" in clone
        assert "https://github.com/jca225/tracklist_engine.git" in clone
        assert "rsync" not in " ".join(cmds)  # GitHub-clone only, never rsync

    def test_bootstrap_skips_clone(self) -> None:
        cmds = [cmd for _, cmd in provision_plan("main", roformer=False)]
        assert any("SKIP_CLONE=1" in c and "vast_bootstrap.sh" in c for c in cmds)

    def test_bashrc_torch_weights_only_export(self) -> None:
        cmds = [cmd for _, cmd in provision_plan("main", roformer=False)]
        assert any(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in c and ".bashrc" in c for c in cmds
        )

    def test_cuda_verify_present(self) -> None:
        cmds = [cmd for _, cmd in provision_plan("main", roformer=False)]
        assert any("torch.cuda.is_available()" in c for c in cmds)

    def test_roformer_optional_and_backgrounded(self) -> None:
        without = [cmd for _, cmd in provision_plan("main", roformer=False)]
        assert not any("roformer" in c for c in without)
        with_ro = [cmd for _, cmd in provision_plan("main", roformer=True)]
        ro = [c for c in with_ro if "setup_roformer_separation.sh" in c]
        assert len(ro) == 1
        assert "nohup" in ro[0] and "&" in ro[0]  # 2.5 GB fetch must not block


class TestAuthorizedKeyLine:
    def test_replaces_comment_with_teardown_tag(self) -> None:
        line = authorized_key_line(
            "ssh-ed25519 AAAAC3Nz root@C.12345", "mert", 40765622
        )
        assert line == "ssh-ed25519 AAAAC3Nz vast-mert-40765622"

    def test_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            authorized_key_line("notakey", "x", 1)


# ---------------------------------------------------------------------------
# Fake-HTTP-driven fetch path
# ---------------------------------------------------------------------------


class TestFetchOffers:
    def test_sends_get_to_bundles_with_query(self) -> None:
        calls: list[tuple[str, str, bytes | None]] = []

        def fake_http(method: str, url: str, body: bytes | None) -> tuple[int, str]:
            calls.append((method, url, body))
            return (
                200,
                '{"offers": [{"id": 7, "direct_port_count": 2, "dph_total": 0.3}]}',
            )

        offers = fetch_offers(fake_http, build_query())
        assert [o.offer_id for o in offers] == [7]
        method, url, body = calls[0]
        assert method == "GET"  # POST is silently rejected by the API
        assert body is None
        assert url.startswith("https://console.vast.ai/api/v0/bundles/?q=")

    def test_non_200_exits(self) -> None:
        def fake_http(method: str, url: str, body: bytes | None) -> tuple[int, str]:
            return 502, "bad gateway"

        with pytest.raises(SystemExit):
            fetch_offers(fake_http, build_query())


class TestTemplateConstant:
    def test_pinned_pytorch_vast_template(self) -> None:
        # 405071 = saved "PyTorch (Vast)" template (/venv/main + sshd).
        assert TEMPLATE_ID == 405071
