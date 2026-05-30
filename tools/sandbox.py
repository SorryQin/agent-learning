import docker
from docker.errors import DockerException

class CodeSandbox:
    """使用Docker容器安全地执行Python代码。"""
    def __init__(
            self, 
            image="python:3.10-slim", 
            memory_limit="128m", 
            timeout=10,
            cpu_limit=0.5,
            pids_limit=64,
            tmpfs_size="64m",
    ):
        try:
            self.client = docker.from_env()
            self.client.ping() # 测试连接
            print("Docker 连接成功！")
        except DockerException as e:
            print(f"Docker 连接失败，请确保 Docker 已安装并运行: {e}")
            exit(1)
        self.image = image
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.tmpfs_size = tmpfs_size

    def run(self, code: str) -> str:
        """在隔离容器中执行代码并返回结果。"""
        container = None
        try:
            container = self.client.containers.run(
                self.image,
                command=["python", "-c", code],
                mem_limit=self.memory_limit,
                network_disabled=True,  # 禁用网络，防止滥用
                read_only=True,         # 只读根文件系统
                detach=True,
                stdout=True,
                stderr=True,
                nano_cpus=int(self.cpu_limit * 1e9), # 0.5 核
                pids_limit=self.pids_limit, # 防 fork 炸弹
                cap_drop=["ALL"], # 剥夺所有特权
                security_opt=["no-new-privileges"], # 禁止提权
                tmpfs={"/tmp": f"size={self.tmpfs_size}, mode=1777"}, # 可写临时盘
            )

            try:
                container.wait(timeout=self.timeout)
            except Exception:
                container.kill()
                return f"代码执行超时（>{self.timeout}秒）"
            logs = container.logs(stdout=True, stderr=True).decode('utf-8', errors='ignore')
            # 可以优化，当输出logs过长时截断（待做）
            return logs if logs else "代码执行完成，无输出。"
        
        except Exception as e:
            return f"代码执行出错: {str(e)}"
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

sandbox = CodeSandbox()