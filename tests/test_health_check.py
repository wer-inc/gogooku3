#!/usr/bin/env python3
"""
Unit tests for health check module.
Tests the health check endpoints and monitoring functionality.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ops.health_check import HealthChecker


class TestHealthChecker:
    """Test cases for HealthChecker class."""

    @pytest.fixture
    def health_checker(self):
        """Create HealthChecker instance for testing."""
        return HealthChecker()

    def test_health_check_structure(self, health_checker):
        """Test that health check returns proper structure."""
        result = health_checker.health_check()

        # Check required fields
        assert "status" in result
        assert "timestamp" in result
        assert "checks" in result
        assert "version" in result
        assert "uptime" in result

        # Check status is valid
        assert result["status"] in ["healthy", "unhealthy", "unknown"]

        # Check timestamp format
        assert "T" in result["timestamp"]  # ISO format

    def test_readiness_check_structure(self, health_checker):
        """Test readiness check structure."""
        result = health_checker.readiness_check()

        assert "status" in result
        assert "timestamp" in result
        assert "checks" in result
        assert result["status"] in ["ready", "not ready"]

    def test_liveness_check_structure(self, health_checker):
        """Test liveness check structure."""
        result = health_checker.liveness_check()

        assert "status" in result
        assert "timestamp" in result
        assert "pid" in result
        assert "uptime" in result
        assert result["status"] == "alive"

    @patch("ops.health_check.psutil.virtual_memory")
    def test_check_memory_healthy(self, mock_memory, health_checker):
        """Test memory check when system is healthy."""
        # Mock healthy memory usage (50%)
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.percent = 50.0
        mock_memory.return_value.total = 16 * 1024**3  # 16GB
        mock_memory.return_value.available = 8 * 1024**3  # 8GB
        mock_memory.return_value.used = 8 * 1024**3  # 8GB

        result = health_checker._check_memory()

        assert result["status"] == "healthy"
        assert "Memory usage: 50.0%" in result["message"]
        assert result["details"]["percentage"] == 50.0

    @patch("ops.health_check.psutil.virtual_memory")
    def test_check_memory_warning(self, mock_memory, health_checker):
        """Test memory check when system is in warning state."""
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.percent = 87.0  # Warning threshold

        result = health_checker._check_memory()

        assert result["status"] == "warning"
        assert "Memory usage: 87.0%" in result["message"]

    @patch("ops.health_check.psutil.virtual_memory")
    def test_check_memory_critical(self, mock_memory, health_checker):
        """Test memory check when system is in critical state."""
        mock_memory.return_value = MagicMock()
        mock_memory.return_value.percent = 96.0  # Critical threshold

        result = health_checker._check_memory()

        assert result["status"] == "critical"
        assert "Memory usage: 96.0%" in result["message"]

    @patch("ops.health_check.psutil.cpu_percent")
    def test_check_cpu_healthy(self, mock_cpu, health_checker):
        """Test CPU check when system is healthy."""
        mock_cpu.return_value = 45.0

        result = health_checker._check_cpu()

        assert result["status"] == "healthy"
        assert "CPU usage: 45.0%" in result["message"]

    @patch("ops.health_check.psutil.cpu_percent")
    def test_check_cpu_critical(self, mock_cpu, health_checker):
        """Test CPU check when system is in critical state."""
        mock_cpu.return_value = 96.0

        result = health_checker._check_cpu()

        assert result["status"] == "critical"
        assert "CPU usage: 96.0%" in result["message"]

    @patch("ops.health_check.psutil.disk_usage")
    def test_check_disk_healthy(self, mock_disk, health_checker):
        """Test disk check when system is healthy."""
        mock_disk.return_value = MagicMock()
        mock_disk.return_value.percent = 60.0
        mock_disk.return_value.total = 100 * 1024**3
        mock_disk.return_value.used = 60 * 1024**3
        mock_disk.return_value.free = 40 * 1024**3

        result = health_checker._check_disk()

        assert result["status"] == "healthy"
        assert "Disk usage: 60.0%" in result["message"]
        assert result["details"]["percentage"] == 60.0

    @patch("ops.health_check.psutil.disk_usage")
    def test_check_disk_critical(self, mock_disk, health_checker):
        """Test disk check when system is in critical state."""
        mock_disk.return_value = MagicMock()
        mock_disk.return_value.percent = 96.0

        result = health_checker._check_disk()

        assert result["status"] == "critical"
        assert "Disk usage: 96.0%" in result["message"]

    @patch("ops.health_check.socket.create_connection")
    def test_check_network_healthy(self, mock_socket, health_checker):
        """Test network check when connectivity is available."""
        mock_socket.return_value = MagicMock()

        result = health_checker._check_network()

        assert result["status"] == "healthy"
        assert "Network connectivity OK" in result["message"]

    @patch("ops.health_check.socket.create_connection")
    def test_check_network_unhealthy(self, mock_socket, health_checker):
        """Test network check when connectivity fails."""
        mock_socket.side_effect = OSError("Network unreachable")

        result = health_checker._check_network()

        assert result["status"] == "unhealthy"
        assert "Network check failed" in result["message"]

    def test_check_filesystem_healthy(self, health_checker, tmp_path):
        """Test filesystem check when all paths exist and are writable."""
        # Create mock directory structure
        logs_dir = tmp_path / "logs"
        data_dir = tmp_path / "data"
        output_dir = tmp_path / "output"

        for dir_path in [logs_dir, data_dir, output_dir]:
            dir_path.mkdir()
            test_file = dir_path / "test.txt"
            test_file.write_text("test")

        # Mock the project root
        with patch.object(health_checker, "project_root", tmp_path):
            result = health_checker._check_filesystem()

            assert result["status"] == "healthy"
            assert "All critical paths accessible" in result["message"]

    def test_check_filesystem_missing_paths(self, health_checker, tmp_path):
        """Test filesystem check when critical paths are missing."""
        with patch.object(health_checker, "project_root", tmp_path):
            result = health_checker._check_filesystem()

            assert result["status"] == "unhealthy"
            assert "missing=" in result["message"]

    def test_check_dependencies_healthy(self, health_checker):
        """Test dependency check when all required modules are available."""
        result = health_checker._check_dependencies()

        # Should be healthy if polars, torch, etc. are installed
        # (this depends on the actual environment)
        assert "status" in result
        assert "message" in result

        if result["status"] == "healthy":
            assert "All critical dependencies available" in result["message"]
        else:
            assert "Missing dependencies:" in result["message"]

    def test_get_version(self, health_checker, tmp_path):
        """Test version retrieval from pyproject.toml."""
        # Create mock pyproject.toml
        pyproject_content = """
[project]
version = "1.2.3"
"""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(pyproject_content)

        with patch.object(health_checker, "project_root", tmp_path):
            version = health_checker._get_version()

            assert version == "1.2.3"

    def test_get_uptime(self, health_checker):
        """Test uptime calculation."""
        uptime = health_checker._get_uptime()

        # Should be a string in HH:MM:SS format
        assert isinstance(uptime, str)
        assert ":" in uptime


class TestHealthCheckCLI:
    """Test command-line interface for health checks."""

    @patch("ops.health_check.HealthChecker")
    def test_cli_health_command(self, mock_checker_class, capsys):
        """Test CLI health command."""
        mock_checker = MagicMock()
        mock_checker.health_check.return_value = {
            "status": "healthy",
            "timestamp": "2023-12-01T10:00:00",
            "checks": {"test": {"status": "healthy", "message": "OK"}},
            "version": "1.0.0",
            "uptime": "00:30:00",
        }
        mock_checker_class.return_value = mock_checker

        # Import and run main function
        import sys
        from unittest.mock import patch

        from ops.health_check import main

        with patch.object(sys, "argv", ["health_check.py", "health"]):
            main()

        captured = capsys.readouterr()
        assert "Status: healthy" in captured.out
        assert "Uptime: 00:30:00" in captured.out

    @patch("ops.health_check.HealthChecker")
    def test_cli_json_format(self, mock_checker_class, capsys):
        """Test CLI with JSON output format."""
        mock_checker = MagicMock()
        mock_checker.health_check.return_value = {"status": "healthy"}
        mock_checker_class.return_value = mock_checker

        import sys
        from unittest.mock import patch

        from ops.health_check import main

        with patch.object(
            sys, "argv", ["health_check.py", "health", "--format", "json"]
        ):
            main()

        captured = capsys.readouterr()
        # Should output valid JSON
        json.loads(captured.out.strip())


if __name__ == "__main__":
    pytest.main([__file__])
