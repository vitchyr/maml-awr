import os
import sys
import time
import subprocess
import numpy as np

output_dir = "output/"

def fetch(loop_sleep_secs):
    loop = (loop_sleep_secs != 0)

    instances = np.genfromtxt(inst_file, dtype="str")
    if len(instances.shape) == 1:
        instances = [instances]
	
    print("Instances:")
    print(instances)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    while True:
        for inst in instances:
            id = inst[0]
            if not id.startswith("#"):
                addr = inst[1]
                inst_dir = output_dir + id + "/"
    
                if not os.path.exists(inst_dir):
                    os.mkdir(inst_dir)

                for src_file in src_files:
                    if id.startswith("aws"):
                        cmd = "pscp -i \"xbpeng_aws.ppk\" ec2-user@{:s}:/home/ec2-user/compsci_stuff/maml-awr/output/{:s} {:s}/".format(addr, src_file, inst_dir)
                    elif id.startswith("gce"):
                        cmd = "pscp -i \"xbpeng_gce.ppk\" xbpeng@{:s}:/home/xbpeng/compsci_stuff/maml-awr/output/{:s} {:s}/".format(addr, src_file, inst_dir)
                    else:
                        assert False, "Unsupportd instance {:s}".format(id)

                    print("cmd: " + cmd)
                    subprocess.call(cmd, shell=True)

        if (loop):
            time.sleep(loop_sleep_secs)
        else:
            break

    return
    
def main():
    fetch(0)
    return

if __name__ == '__main__':
    main()