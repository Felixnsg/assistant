#!/usr/bin/env python3
"""
Orpheus TTS Server Health Monitor
Monitors server health, GPU usage, and performance metrics
"""

import asyncio
import aiohttp
import time
import psutil
import torch
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HealthCheck:
    """Health check result"""
    timestamp: str
    status: str
    latency_ms: float
    model_loaded: bool
    gpu_available: Optional[bool] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[int] = None
    error: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics from server"""
    timestamp: str
    requests_total: int
    requests_per_second: float
    ttfb_ms_avg: float
    ttfb_ms_p95: float
    ttfb_ms_p99: float
    rtf_avg: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: float

class OrpheusMonitor:
    """Monitor for Orpheus TTS server"""
    
    def __init__(self, server_url: str, check_interval: int = 30):
        self.server_url = server_url.rstrip('/')
        self.check_interval = check_interval
        self.health_history: List[HealthCheck] = []
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_thresholds = {
            "max_ttfb_ms": 500,
            "max_rtf": 1.0,
            "max_gpu_memory_percent": 90,
            "max_cpu_percent": 80,
            "max_memory_percent": 80,
            "min_requests_per_second": 0.1,
        }
        self.alert_cooldown = {}
        self.test_texts = [
            "Hello, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing server performance and reliability.",
        ]
    
    async def check_health(self) -> HealthCheck:
        """Check server health"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        return HealthCheck(
                            timestamp=datetime.now().isoformat(),
                            status="healthy",
                            latency_ms=latency_ms,
                            model_loaded=data.get("model_loaded", False),
                            gpu_available=data.get("cuda_available"),
                            gpu_memory_used_gb=data.get("gpu", {}).get("memory_allocated_mb", 0) / 1024,
                            gpu_memory_total_gb=data.get("gpu", {}).get("memory_total_gb"),
                            gpu_utilization=data.get("gpu", {}).get("utilization"),
                        )
                    else:
                        return HealthCheck(
                            timestamp=datetime.now().isoformat(),
                            status="unhealthy",
                            latency_ms=latency_ms,
                            model_loaded=False,
                            error=f"HTTP {response.status}"
                        )
                        
        except asyncio.TimeoutError:
            return HealthCheck(
                timestamp=datetime.now().isoformat(),
                status="timeout",
                latency_ms=(time.time() - start_time) * 1000,
                model_loaded=False,
                error="Request timeout"
            )
        except Exception as e:
            return HealthCheck(
                timestamp=datetime.now().isoformat(),
                status="error",
                latency_ms=(time.time() - start_time) * 1000,
                model_loaded=False,
                error=str(e)
            )
    
    async def get_metrics(self) -> Optional[PerformanceMetrics]:
        """Get performance metrics from server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/stats",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return PerformanceMetrics(
                            timestamp=datetime.now().isoformat(),
                            requests_total=data.get("requests_total", 0),
                            requests_per_second=data.get("requests_per_second", 0),
                            ttfb_ms_avg=data.get("ttfb_ms", {}).get("avg", 0),
                            ttfb_ms_p95=data.get("ttfb_ms", {}).get("p95", 0),
                            ttfb_ms_p99=data.get("ttfb_ms", {}).get("p99", 0),
                            rtf_avg=data.get("rtf", {}).get("avg", 0),
                            cpu_percent=data.get("system", {}).get("cpu_percent", 0),
                            memory_percent=data.get("system", {}).get("memory_percent", 0),
                            gpu_memory_mb=data.get("system", {}).get("gpu_memory_mb", 0),
                        )
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return None
    
    async def test_generation(self) -> Dict:
        """Test actual TTS generation"""
        results = []
        
        for text in self.test_texts:
            start_time = time.time()
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.server_url}/tts",
                        json={
                            "text": text,
                            "voice": "tara",
                            "stream": False
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            content = await response.read()
                            duration = time.time() - start_time
                            
                            results.append({
                                "text_length": len(text),
                                "audio_size": len(content),
                                "generation_time": duration,
                                "success": True
                            })
                        else:
                            results.append({
                                "text_length": len(text),
                                "error": f"HTTP {response.status}",
                                "success": False
                            })
                            
            except Exception as e:
                results.append({
                    "text_length": len(text),
                    "error": str(e),
                    "success": False
                })
        
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
        avg_time = statistics.mean([r["generation_time"] for r in results if "generation_time" in r] or [0])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "success_rate": success_rate,
            "average_generation_time": avg_time,
            "tests": results
        }
    
    def check_alerts(self, health: HealthCheck, metrics: Optional[PerformanceMetrics]):
        """Check for alert conditions"""
        alerts = []
        current_time = time.time()
        
        # Health alerts
        if health.status != "healthy":
            alert_key = "health_status"
            if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                alerts.append(f"🚨 Server is {health.status}: {health.error}")
                self.alert_cooldown[alert_key] = current_time
        
        if health.latency_ms > 1000:
            alert_key = "high_latency"
            if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                alerts.append(f"⚠️ High health check latency: {health.latency_ms:.1f}ms")
                self.alert_cooldown[alert_key] = current_time
        
        # Metrics alerts
        if metrics:
            if metrics.ttfb_ms_avg > self.alert_thresholds["max_ttfb_ms"]:
                alert_key = "high_ttfb"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                    alerts.append(f"⚠️ High TTFB: {metrics.ttfb_ms_avg:.1f}ms (threshold: {self.alert_thresholds['max_ttfb_ms']}ms)")
                    self.alert_cooldown[alert_key] = current_time
            
            if metrics.rtf_avg > self.alert_thresholds["max_rtf"]:
                alert_key = "high_rtf"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                    alerts.append(f"⚠️ High RTF: {metrics.rtf_avg:.2f}x (threshold: {self.alert_thresholds['max_rtf']}x)")
                    self.alert_cooldown[alert_key] = current_time
            
            if metrics.cpu_percent > self.alert_thresholds["max_cpu_percent"]:
                alert_key = "high_cpu"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                    alerts.append(f"⚠️ High CPU usage: {metrics.cpu_percent:.1f}%")
                    self.alert_cooldown[alert_key] = current_time
            
            if metrics.memory_percent > self.alert_thresholds["max_memory_percent"]:
                alert_key = "high_memory"
                if alert_key not in self.alert_cooldown or current_time - self.alert_cooldown[alert_key] > 300:
                    alerts.append(f"⚠️ High memory usage: {metrics.memory_percent:.1f}%")
                    self.alert_cooldown[alert_key] = current_time
        
        return alerts
    
    def print_status(self, health: HealthCheck, metrics: Optional[PerformanceMetrics]):
        """Print current status"""
        print("\n" + "=" * 60)
        print(f"ORPHEUS TTS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Health status
        status_emoji = "✅" if health.status == "healthy" else "❌"
        print(f"\n{status_emoji} Status: {health.status.upper()}")
        print(f"   Latency: {health.latency_ms:.1f}ms")
        print(f"   Model Loaded: {health.model_loaded}")
        
        if health.gpu_available:
            print(f"\n🖥️ GPU Status:")
            print(f"   Memory: {health.gpu_memory_used_gb:.1f}/{health.gpu_memory_total_gb:.1f} GB")
            if health.gpu_utilization:
                print(f"   Utilization: {health.gpu_utilization}%")
        
        # Performance metrics
        if metrics:
            print(f"\n📊 Performance:")
            print(f"   Requests: {metrics.requests_total} total")
            print(f"   Throughput: {metrics.requests_per_second:.2f} req/s")
            print(f"   TTFB: {metrics.ttfb_ms_avg:.1f}ms avg, {metrics.ttfb_ms_p95:.1f}ms p95, {metrics.ttfb_ms_p99:.1f}ms p99")
            print(f"   RTF: {metrics.rtf_avg:.2f}x")
            print(f"\n💻 System:")
            print(f"   CPU: {metrics.cpu_percent:.1f}%")
            print(f"   Memory: {metrics.memory_percent:.1f}%")
            print(f"   GPU Memory: {metrics.gpu_memory_mb:.1f} MB")
        
        # History summary
        if len(self.health_history) > 1:
            recent_health = self.health_history[-10:]
            uptime_percent = sum(1 for h in recent_health if h.status == "healthy") / len(recent_health) * 100
            avg_latency = statistics.mean([h.latency_ms for h in recent_health])
            
            print(f"\n📈 Recent History (last {len(recent_health)} checks):")
            print(f"   Uptime: {uptime_percent:.1f}%")
            print(f"   Avg Latency: {avg_latency:.1f}ms")
    
    async def run_monitor(self):
        """Main monitoring loop"""
        logger.info(f"Starting monitor for {self.server_url}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        while True:
            try:
                # Run health check
                health = await self.check_health()
                self.health_history.append(health)
                
                # Get metrics
                metrics = await self.get_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                
                # Check for alerts
                alerts = self.check_alerts(health, metrics)
                for alert in alerts:
                    logger.warning(alert)
                
                # Print status
                self.print_status(health, metrics)
                
                # Save history (keep last 1000 entries)
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Save to file periodically
                if len(self.health_history) % 10 == 0:
                    self.save_history()
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            # Wait for next check
            await asyncio.sleep(self.check_interval)
    
    async def run_test(self):
        """Run a single test of all endpoints"""
        print("\n" + "=" * 60)
        print("ORPHEUS TTS SERVER TEST")
        print("=" * 60)
        
        # Health check
        print("\n🔍 Testing /health endpoint...")
        health = await self.check_health()
        print(f"   Status: {health.status}")
        print(f"   Latency: {health.latency_ms:.1f}ms")
        print(f"   Model Loaded: {health.model_loaded}")
        
        # Metrics
        print("\n📊 Testing /stats endpoint...")
        metrics = await self.get_metrics()
        if metrics:
            print(f"   Requests: {metrics.requests_total}")
            print(f"   TTFB Avg: {metrics.ttfb_ms_avg:.1f}ms")
            print(f"   RTF: {metrics.rtf_avg:.2f}x")
        else:
            print("   ❌ Failed to get metrics")
        
        # Generation test
        print("\n🎤 Testing TTS generation...")
        test_results = await self.test_generation()
        print(f"   Success Rate: {test_results['success_rate']:.1f}%")
        print(f"   Avg Time: {test_results['average_generation_time']:.2f}s")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
    
    def save_history(self):
        """Save monitoring history to file"""
        try:
            with open("monitor_history.json", "w") as f:
                json.dump({
                    "health": [asdict(h) for h in self.health_history[-100:]],
                    "metrics": [asdict(m) for m in self.metrics_history[-100:] if m]
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Orpheus TTS Server Monitor")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--test", action="store_true", help="Run single test and exit")
    
    args = parser.parse_args()
    
    monitor = OrpheusMonitor(args.url, args.interval)
    
    if args.test:
        await monitor.run_test()
    else:
        await monitor.run_monitor()

if __name__ == "__main__":
    asyncio.run(main())