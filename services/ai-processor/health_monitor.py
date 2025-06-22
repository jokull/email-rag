"""
Health Monitor for AI Processor
Monitors system health, memory usage, and performance
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from llm_interface import QwenInterface
from email_processor import EmailProcessor
from embedding_service import EmbeddingService

class HealthMonitor:
    """
    Monitors the health of all AI services and system resources
    Optimized for Mac mini M2 16GB RAM constraints
    """
    
    def __init__(
        self,
        llm_interface: QwenInterface,
        email_processor: EmailProcessor,
        embedding_service: EmbeddingService
    ):
        self.llm = llm_interface
        self.processor = email_processor
        self.embeddings = embedding_service
        self.logger = logging.getLogger(__name__)
        
        # Health tracking
        self.start_time = datetime.utcnow()
        self.last_health_check = None
        self.health_history = []
        self.max_history_size = 100
        
        # Memory thresholds for Mac mini M2 16GB
        self.memory_warning_threshold = 0.8  # 80% of available RAM
        self.memory_critical_threshold = 0.9  # 90% of available RAM
        
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the AI processor"""
        current_time = datetime.utcnow()
        
        try:
            # System metrics
            system_health = await self._get_system_health()
            
            # Service health checks
            llm_healthy = await self.llm.health_check() if self.llm else False
            embeddings_healthy = await self.embeddings.health_check() if self.embeddings else False
            processor_healthy = not self.processor.is_processing if self.processor else True
            
            # Overall health determination
            overall_healthy = (
                system_health["memory_healthy"] and
                system_health["cpu_healthy"] and
                llm_healthy and
                embeddings_healthy and
                processor_healthy
            )
            
            health_status = {
                "healthy": overall_healthy,
                "timestamp": current_time,
                "uptime_seconds": (current_time - self.start_time).total_seconds(),
                "services": {
                    "llm_interface": llm_healthy,
                    "embedding_service": embeddings_healthy,
                    "email_processor": processor_healthy
                },
                "system": system_health,
                "warnings": self._get_warnings(system_health)
            }
            
            # Store in history
            self._update_health_history(health_status)
            self.last_health_check = current_time
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": current_time
            }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system resource health metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100
            memory_healthy = memory_percent < self.memory_warning_threshold
            
            # CPU usage (average over last second)
            cpu_percent = psutil.cpu_percent(interval=1) / 100
            cpu_healthy = cpu_percent < 0.8  # Less than 80% CPU
            
            # Disk usage for logs and models
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100
            disk_healthy = disk_percent < 0.9  # Less than 90% disk
            
            # Get process-specific memory usage
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            return {
                "memory_healthy": memory_healthy,
                "cpu_healthy": cpu_healthy,
                "disk_healthy": disk_healthy,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 1),
                    "available_gb": round(memory.available / (1024**3), 1),
                    "used_percent": round(memory_percent * 100, 1),
                    "process_memory_mb": round(process_memory, 1)
                },
                "cpu": {
                    "usage_percent": round(cpu_percent * 100, 1),
                    "cores": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 1),
                    "free_gb": round(disk.free / (1024**3), 1),
                    "used_percent": round(disk_percent * 100, 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ System health check failed: {e}")
            return {
                "memory_healthy": False,
                "cpu_healthy": False,
                "disk_healthy": False,
                "error": str(e)
            }
    
    def _get_warnings(self, system_health: Dict[str, Any]) -> List[str]:
        """Generate warning messages based on system health"""
        warnings = []
        
        if not system_health.get("memory_healthy"):
            memory_percent = system_health.get("memory", {}).get("used_percent", 0)
            warnings.append(f"High memory usage: {memory_percent:.1f}%")
        
        if not system_health.get("cpu_healthy"):
            cpu_percent = system_health.get("cpu", {}).get("usage_percent", 0)
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if not system_health.get("disk_healthy"):
            disk_percent = system_health.get("disk", {}).get("used_percent", 0)
            warnings.append(f"High disk usage: {disk_percent:.1f}%")
        
        # Check process memory specifically for Qwen-0.5B
        process_memory = system_health.get("memory", {}).get("process_memory_mb", 0)
        if process_memory > 3000:  # More than 3GB for our process
            warnings.append(f"High process memory usage: {process_memory:.0f}MB")
        
        return warnings
    
    def _update_health_history(self, health_status: Dict[str, Any]):
        """Update health history for trend analysis"""
        self.health_history.append({
            "timestamp": health_status["timestamp"],
            "healthy": health_status["healthy"],
            "memory_percent": health_status.get("system", {}).get("memory", {}).get("used_percent", 0),
            "cpu_percent": health_status.get("system", {}).get("cpu", {}).get("usage_percent", 0)
        })
        
        # Keep only recent history
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for all services"""
        try:
            # Get health status
            health_status = await self.get_health_status()
            
            # Get LLM metrics
            llm_metrics = self.llm.get_performance_metrics() if self.llm else {}
            
            # Get processor metrics
            processor_status = await self.processor.get_processing_status() if self.processor else {}
            
            # Get embedding model info
            embedding_info = self.embeddings.get_model_info() if self.embeddings else {}
            
            # Calculate health trends
            health_trends = self._calculate_health_trends()
            
            return {
                "health": health_status,
                "llm": llm_metrics,
                "processor": processor_status,
                "embeddings": embedding_info,
                "trends": health_trends,
                "recommendations": self._get_recommendations(health_status)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get detailed metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_health_trends(self) -> Dict[str, Any]:
        """Calculate health trends from history"""
        if len(self.health_history) < 2:
            return {"insufficient_data": True}
        
        # Get recent data points
        recent = self.health_history[-10:]  # Last 10 checks
        
        # Calculate averages
        avg_memory = sum(h["memory_percent"] for h in recent) / len(recent)
        avg_cpu = sum(h["cpu_percent"] for h in recent) / len(recent)
        health_rate = sum(1 for h in recent if h["healthy"]) / len(recent)
        
        # Calculate trends (simple slope)
        if len(recent) >= 3:
            memory_trend = recent[-1]["memory_percent"] - recent[-3]["memory_percent"]
            cpu_trend = recent[-1]["cpu_percent"] - recent[-3]["cpu_percent"]
        else:
            memory_trend = 0
            cpu_trend = 0
        
        return {
            "averages": {
                "memory_percent": round(avg_memory, 1),
                "cpu_percent": round(avg_cpu, 1),
                "health_rate": round(health_rate * 100, 1)
            },
            "trends": {
                "memory_trend": round(memory_trend, 1),
                "cpu_trend": round(cpu_trend, 1),
                "memory_direction": "increasing" if memory_trend > 1 else "decreasing" if memory_trend < -1 else "stable",
                "cpu_direction": "increasing" if cpu_trend > 5 else "decreasing" if cpu_trend < -5 else "stable"
            }
        }
    
    def _get_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Get recommendations based on current health status"""
        recommendations = []
        
        system = health_status.get("system", {})
        memory = system.get("memory", {})
        cpu = system.get("cpu", {})
        
        # Memory recommendations
        memory_percent = memory.get("used_percent", 0)
        if memory_percent > 85:
            recommendations.append("Consider reducing batch size or processing interval to lower memory usage")
        elif memory_percent > 75:
            recommendations.append("Monitor memory usage closely - approaching capacity")
        
        # CPU recommendations  
        cpu_percent = cpu.get("usage_percent", 0)
        if cpu_percent > 80:
            recommendations.append("High CPU usage detected - consider reducing concurrent processing")
        
        # Process memory recommendations
        process_memory = memory.get("process_memory_mb", 0)
        if process_memory > 2500:  # More than 2.5GB for our process
            recommendations.append("AI processor using high memory - restart may help reclaim memory")
        
        # Service-specific recommendations
        if not health_status.get("services", {}).get("llm_interface"):
            recommendations.append("LLM interface unhealthy - check Qwen model loading")
        
        if not health_status.get("services", {}).get("embedding_service"):
            recommendations.append("Embedding service unhealthy - check sentence-transformers model")
        
        # General recommendations for Mac mini M2
        if not recommendations:
            recommendations.append("System running optimally for Mac mini M2 16GB configuration")
        
        return recommendations
    
    async def emergency_shutdown_check(self) -> bool:
        """Check if emergency shutdown is needed due to resource constraints"""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100
            
            # Emergency shutdown if memory usage is critical
            if memory_percent > self.memory_critical_threshold:
                self.logger.critical(f"🚨 CRITICAL: Memory usage at {memory_percent*100:.1f}% - emergency shutdown recommended")
                return True
            
            # Check if system is completely unresponsive
            try:
                start_time = time.time()
                await asyncio.sleep(0.1)
                response_time = time.time() - start_time
                
                if response_time > 1.0:  # System very slow
                    self.logger.warning("⚠️ System response time degraded")
                    
            except Exception:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Emergency shutdown check failed: {e}")
            return True  # Fail safe