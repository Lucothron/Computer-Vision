import os, string

def interface (name, input):
    '''
    Interface module between FACT and the cbir program.
    '''
    # Run the program, saving the result into tempfile.  Read that file and
    # from it determine which category the program has decided input belongs to,
    # then delete tempfile.  If anything goes wrong, return a failure.
    tempfile = 'RESULT'
    cmd = './cbir3s %s *.png > %s' % (input, tempfile)
    try:
        os.system (cmd)
        fd = open (tempfile, 'r')
        category = fd.readline().split()[0]
        fd.close ()
        os.remove (tempfile)
        i = category.find ('-')
        result = category[0:i]
        status = True
    except:
        result = 'failure'
        status = False
    return status, result
