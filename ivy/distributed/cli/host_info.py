class HostInfo:
    def __init__(self):
        self.nodes = []
        self.ssh_port = 22

    def load_from_host_str(self, host_str):
        self.nodes = host_str.split(",")

    def load_from_hostfile(self, hostfile_path):
        with open(hostfile_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line[0] == "#":
                    continue
                if "#" in line:
                    line = line.split("#")[0]
                host = line.split(" ")[0]
                self.nodes.append(host)

    def __iter__(self):
        return (host for host in self.nodes)
