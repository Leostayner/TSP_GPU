import subprocess
import sys 

dirExe    = "../build/" 
exe       = ["2-opt-sol"] 
           
inputs = ["input1"]
output = ["output1"]

dir_inp = "../inputs/"
dir_out = "../outputs/"

def check_output(exe):
    print(exe)
    for input_file, outout_file in zip(inputs, output):
        command = dirExe + exe + " < " + dir_inp + input_file
        stdout  = subprocess.check_output(command, shell=True, stderr=subprocess.DEVNULL).decode(sys.stdout.encoding)
        
        with open(dir_out + outout_file, 'r') as f:
            outout_file = f.read()
        
        st = (stdout.split("\n"))
        ot = (outout_file.split("\n"))
        
        c1 = st[0]
        s1 = st[1].split(" ")

        c2 = ot[0]
        s2 = ot[1].split(" ")
        
        if(c1 == c2 and (s1 == s2 or s1 == ([s2[0]] + s2[:0:-1])) ):
            print(input_file, ": ", u'\u2713')

        else:
            print(input_file, ": ", "x")
            

for e in exe:
    check_output(e)
    print()
