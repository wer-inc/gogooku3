#!/usr/bin/env python3
"""
End-to-End tests for gogooku3-standalone Docker environment.
Tests the complete application workflow in a containerized environment.
"""

import os
import sys
import time
import json
import pytest
import requests
import subprocess
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestE2EDocker:
    """End-to-end tests for Docker-based deployment."""

    def setup_method(self):
        """Setup for each test method."""
        self.project_root = Path(__file__).parent.parent
        self.compose_file = self.project_root / "docker-compose.yml"
        self.override_file = self.project_root / "docker-compose.override.yml"

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up any test artifacts
        pass

    @pytest.mark.slow
    def test_docker_services_start(self):
        """Test that all Docker services can start successfully."""
        if not self.compose_file.exists():
            pytest.skip("docker-compose.yml not found")

        try:
            # Start services
            result = subprocess.run(
                ["docker", "compose", "up", "-d"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )

            assert result.returncode == 0, f"Docker compose failed: {result.stderr}"

            # Wait for services to be healthy
            time.sleep(30)

            # Check service status
            result = subprocess.run(
                ["docker", "compose", "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            assert result.returncode == 0
            assert "Up" in result.stdout or "running" in result.stdout

        finally:
            # Clean up
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True
            )

    def test_docker_compose_config_valid(self):
        """Test that docker-compose configuration is valid."""
        if not self.compose_file.exists():
            pytest.skip("docker-compose.yml not found")

        result = subprocess.run(
            ["docker", "compose", "config"],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Invalid compose config: {result.stderr}"

    def test_environment_variables_configured(self):
        """Test that required environment variables are properly configured."""
        env_example = self.project_root / ".env.example"

        if not env_example.exists():
            pytest.skip(".env.example not found")

        # Read .env.example
        with open(env_example, 'r') as f:
            content = f.read()

        # Check for required environment variables
        required_vars = [
            "MINIO_ROOT_USER",
            "MINIO_ROOT_PASSWORD",
            "CLICKHOUSE_USER",
            "CLICKHOUSE_PASSWORD",
            "REDIS_PASSWORD"
        ]

        for var in required_vars:
            assert var in content, f"Required variable {var} not found in .env.example"

    @pytest.mark.integration
    def test_minio_service_health(self):
        """Test MinIO service health endpoint."""
        if not self.compose_file.exists():
            pytest.skip("Docker environment not available")

        try:
            # Start services
            subprocess.run(
                ["docker", "compose", "up", "-d", "minio"],
                cwd=self.project_root,
                check=True
            )

            # Wait for MinIO to start
            time.sleep(10)

            # Test MinIO health endpoint
            response = requests.get("http://localhost:9000/minio/health/live", timeout=5)
            assert response.status_code == 200

        finally:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True
            )

    @pytest.mark.integration
    def test_clickhouse_service_health(self):
        """Test ClickHouse service health."""
        if not self.compose_file.exists():
            pytest.skip("Docker environment not available")

        try:
            # Start services
            subprocess.run(
                ["docker", "compose", "up", "-d", "clickhouse"],
                cwd=self.project_root,
                check=True
            )

            # Wait for ClickHouse to start
            time.sleep(15)

            # Test ClickHouse connection
            result = subprocess.run([
                "docker", "exec", "gogooku3-clickhouse",
                "clickhouse-client", "--query", "SELECT 1"
            ], capture_output=True, text=True)

            assert result.returncode == 0
            assert "1" in result.stdout

        finally:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True
            )

    @pytest.mark.integration
    def test_redis_service_health(self):
        """Test Redis service health."""
        if not self.compose_file.exists():
            pytest.skip("Docker environment not available")

        try:
            # Start services
            subprocess.run(
                ["docker", "compose", "up", "-d", "redis"],
                cwd=self.project_root,
                check=True
            )

            # Wait for Redis to start
            time.sleep(5)

            # Test Redis connection
            result = subprocess.run([
                "docker", "exec", "gogooku3-redis",
                "redis-cli", "ping"
            ], capture_output=True, text=True)

            assert result.returncode == 0
            assert "PONG" in result.stdout

        finally:
            subprocess.run(
                ["docker", "compose", "down"],
                cwd=self.project_root,
                capture_output=True
            )

    def test_application_startup(self):
        """Test that the main application can start."""
        # Test that main.py exists and is executable
        main_py = self.project_root / "main.py"
        assert main_py.exists(), "main.py not found"
        assert main_py.stat().st_mode & 0o111, "main.py is not executable"

        # Test that we can import the main module
        sys.path.insert(0, str(self.project_root))
        try:
            import main
            assert hasattr(main, 'main'), "main.py does not have main function"
        except ImportError as e:
            pytest.skip(f"Cannot import main module: {e}")

    def test_health_check_endpoint(self):
        """Test health check endpoint functionality."""
        from ops.health_check import HealthChecker

        checker = HealthChecker()
        result = checker.health_check()

        # Verify response structure
        assert isinstance(result, dict)
        assert "status" in result
        assert "timestamp" in result
        assert "checks" in result

        # Verify health status
        assert result["status"] in ["healthy", "unhealthy", "unknown"]

    def test_metrics_endpoint(self):
        """Test metrics endpoint functionality."""
        from ops.metrics_exporter import MetricsExporter

        exporter = MetricsExporter()
        metrics = exporter.generate_metrics()

        # Verify metrics format
        assert isinstance(metrics, str)
        assert len(metrics) > 0

        # Check for key metrics
        assert "gogooku3_uptime_seconds" in metrics
        assert "gogooku3_memory_usage_percent" in metrics
        assert "gogooku3_cpu_usage_percent" in metrics

    def test_configuration_files_exist(self):
        """Test that all required configuration files exist."""
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            "README.md",
            ".gitignore"
        ]

        for filename in required_files:
            file_path = self.project_root / filename
            assert file_path.exists(), f"Required file {filename} not found"

    def test_security_configuration(self):
        """Test security-related configuration."""
        gitignore = self.project_root / ".gitignore"

        if gitignore.exists():
            with open(gitignore, 'r') as f:
                content = f.read()

            # Check that sensitive files are ignored
            assert ".env" in content, ".env should be in .gitignore"
            assert "!.env.example" in content, ".env.example should be tracked"

    def test_logging_configuration(self):
        """Test logging directory and configuration."""
        logs_dir = self.project_root / "logs"

        # Logs directory should exist or be creatable
        if logs_dir.exists():
            assert logs_dir.is_dir(), "logs should be a directory"
        else:
            # Check if main.py can create it
            pass  # This would require running main.py

    @pytest.mark.parametrize("service_name,expected_image", [
        ("minio", "minio/minio:latest"),
        ("clickhouse", "clickhouse/clickhouse-server:latest"),
        ("redis", "redis:7-alpine"),
    ])
    def test_docker_service_images(self, service_name, expected_image):
        """Test that Docker services use expected images."""
        if not self.compose_file.exists():
            pytest.skip("docker-compose.yml not found")

        with open(self.compose_file, 'r') as f:
            content = f.read()

        assert f"image: {expected_image}" in content, \
            f"Service {service_name} should use image {expected_image}"

    def test_environment_file_structure(self):
        """Test .env.example file structure."""
        env_example = self.project_root / ".env.example"

        if not env_example.exists():
            pytest.skip(".env.example not found")

        with open(env_example, 'r') as f:
            content = f.read()

        # Check for required sections
        required_sections = [
            "MinIO",
            "ClickHouse",
            "Redis"
        ]

        for section in required_sections:
            assert section.upper() in content.upper(), \
                f"Section {section} not found in .env.example"


class TestE2EWorkflow:
    """Test complete application workflows."""

    def test_safe_training_workflow_help(self):
        """Test that safe-training workflow help works."""
        main_py = Path(__file__).parent.parent / "main.py"

        if not main_py.exists():
            pytest.skip("main.py not found")

        result = subprocess.run([
            sys.executable, str(main_py), "--help"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert "safe-training" in result.stdout
        assert "ml-dataset" in result.stdout

    def test_safe_training_workflow_validation(self):
        """Test safe-training workflow argument validation."""
        main_py = Path(__file__).parent.parent / "main.py"

        if not main_py.exists():
            pytest.skip("main.py not found")

        # Test invalid workflow
        result = subprocess.run([
            sys.executable, str(main_py), "invalid-workflow"
        ], capture_output=True, text=True)

        assert result.returncode != 0  # Should fail with invalid workflow

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires full environment setup")
    def test_complete_training_workflow(self):
        """Test complete training workflow (requires full environment)."""
        # This test would require:
        # 1. Full Docker environment
        # 2. Sample data
        # 3. All dependencies
        # For now, skip this integration test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
