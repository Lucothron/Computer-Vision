#!/usr/bin/env python3
'''FACT: Framework for Algorithm Compaarison and Testing

FACT is a test harness intended to be used for evaluating the performance
of computer vision algorithms, both in isolation and in comparison with
other algorithms.

Usage:
    fact help

    fact review <test-file>

    fact execute <test-file>
    fact run     <test-file>

    fact vary "<par>=<val>,<val>"...  <test-file>
    fact roc  "<par>=<val>,<val>"...  <test-file>

    fact optimise "<par>=<min>,<max>"...  <test-file>
    fact optimize "<par>=<min>,<max>"...  <test-file>
    fact opt      "<par>=<min>,<max>"...  <test-file>

    fact analyse <transcript>
    fact analyze <transcript>
    fact anal    <transcript>

    fact compare <transcript> <transcript>...
'''

#-----------------------------------------------------------------------------
# REVISION HISTORY
# 0.00 2009-01-28 Started coding.
#-----------------------------------------------------------------------------

import sys, optparse, os, math, time

#-----------------------------------------------------------------------------
# G L O B A L    V A R I A B L E S
#-----------------------------------------------------------------------------
timestamp = "Time-stamp: <2020-10-13 08:57:43 Adrian F Clark (alien@essex.ac.uk)>"
reportnargs = {'result': 4, 'transcript_begin': 5, 'transcript_end': 3}
scriptnargs = {'author': 2, 'name': 1, 'purpose': 1, 'test': 3, 'tests': 1,
         'type': 1, 'url': 1, 'version': 1}

preamble = {'html': r'''<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"> 
<HEAD>
<SCRIPT TYPE="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</SCRIPT>
<SCRIPT TYPE="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</SCRIPT>
<LINK HREF='jquiz.css' TYPE='text/css' REL='stylesheet' />
<SCRIPT TYPE="text/javascript"
 SRC="http://ajax.googleapis.com/ajax/libs/jquery/1.5.2/jquery.min.js"></SCRIPT>
<SCRIPT TYPE="text/javascript" SRC="jquiz.js"></SCRIPT>
</HEAD>
<BODY>''',
            'latex': '\\documentclass[a4paper]{article}\n' + \
            '\\usepackage[T1]{fontenc}\n' + \
            '\\usepackage{a4wide,alien,alienfonts}\n' + \
            '\\begin{document}',
            'text': ''}

postamble = {'html': '</BODY>\n</HTML>',
             'latex': '\\end{document}',
             'text': ''}

# The following formats are used by format_error_rates.
error_rate_format = {}
error_rate_format['header,html'] = r'''
<!-- Calculated from %s -->
<TABLE ALIGN="center" BORDER="1" CELLPADDING="3">
<TR>
  <TH>tests</TH><TH>TP</TH><TH>TN</TH><TH>FP</TH>
  <TH>FN</TH><TH>accuracy</TH><TH>recall</TH><TH>precision</TH>
  <TH>specificity</TH><TH>class</TH>
</TR>
'''

error_rate_format['header,latex'] = r'''
\begin{table}
  %% Calculated from %s
  \begin{center}
    \begin{tabular}{rrrrrrrrrc}
      \toprule
      tests & TP & TN & FP & FN & accuracy & recall & precision &
      specificity & class \\
      \midrule
'''

error_rate_format['header,text'] = 'Error rates calculated from %s\n' \
                                    + '#  tests      TP      TN      FP'\
                                    + '      FN accuracy   recall precision'\
                                    + ' specificity class\n'

error_rate_format['layout,html'] =  r'''
<TR>
  <TD ALIGN="right"> %d </TD><TD ALIGN="right"> %d </TD>
  <TD ALIGN="right"> %d </TD><TD ALIGN="right"> %d </TD>
  <TD ALIGN="right"> %d </TD><TD ALIGN="right"> %.2f </TD>
  <TD ALIGN="right"> %.2f </TD><TD ALIGN="right"> %.2f </TD>
  <TD ALIGN="right"> %.2f </TD><TD ALIGN="center"> <B>%s</B> </TD>
</TR>
'''

error_rate_format['layout,latex'] = '      %d & %d & %d & %d & %d & %.2f' \
                                    + '& %.2f & %.2f & %.2f & \\textbf{%s} ' \
                                    + '\\\\\n'

error_rate_format['layout,text'] = r'''%8d %7d %7d %7d %7d %8.2f %8.2f %9.2f %11.2f %s
'''

error_rate_format['footer,html'] = r'''</TABLE>
</P>

<P ALIGN="center">Error rates calculated from %s</P>'''

error_rate_format['separator,html'] = ''
error_rate_format['separator,latex'] = r''
error_rate_format['separator,text'] = ''

error_rate_format['footer,latex'] = r'''      \bottomrule
   \end{tabular}
  \end{center}
  \caption{Error rates calculated from \texttt{%s}}
  \label{ertab:%s}
\end{table}
'''

error_rate_format['footer,text'] = ''

error_rate_format['detail2,html'] = r'''
<P> The table shown above summarizes the performance of the algorithm on the
tests in <TT> %s </TT> as calculated by
<A HREF="http://fact.essex.ac.uk">FACT</A>,
where TP is the number of true positives <EM>etc</EM>, <EM>N</EM> is the
number of tests, and
$$\mbox{accuracy} = \frac{\mbox{TP} + \mbox{TN}}{N}$$
$$\mbox{precision} = \frac{\mbox{TP}}{\mbox{TP} + \mbox{FP}}$$
$$\mbox{sensitivity} = \mbox{recall} = \frac{\mbox{TP}}
    {\mbox{TP} + \mbox{FN}}$$
$$\mbox{specificity} = \frac{\mbox{TN}}{\mbox{FP} + \mbox{TN}}$$
'''

error_rate_format['detail2,latex'] = r'''Table~\ref{ertab:%s} sumamrizes the
performance of the algorithm on the tests in \texttt{%s},
as calculated by FACT,\footnote{See \url{http://fact.essex.ac.uk/}.}
where TP is the number of true positives \emph{etc}, $N$ is the number of
tests, and
\begin{eqnarray}
  \mbox{accuracy} &=& \frac{\mbox{TP} + \mbox{TN}}{N}\\
  \mbox{precision} &=& \frac{\mbox{TP}}{\mbox{TP} + \mbox{FP}}\\
  \mbox{sensitivity} \equiv \mbox{recall} &=& \frac{\mbox{TP}}
    {\mbox{TP} + \mbox{FN}}\\
  \mbox{specificity} &=& \frac{\mbox{TN}}{\mbox{FP} + \mbox{TN}}
  \label{eq:%s}
\end{eqnarray}'''

error_rate_format['detail2,text'] = r'''
The table shown above summarizes the performance of the algorithm on
the tests in %s as calculated by FACT (http://fact.essex.ac.uk/).
Here, TP is the number of true positives etc, N is the number of tests, and

              TP + TN                              TP
   accuracy = -------                precision = -------
                 N                               TP + FP

                            TP                     TN
  sensitivity = recall = -------   specificity = -------
                         TP + FN                 FP + TN

'''

confmat_format = {}
confmat_format['detail2,html'] = r'''
<P>The above table gives the corresponding class confusion matrix.
Here, the element at row <EM>r</EM> and column <EM>c</EM> indicates
that class <EM>r</EM> was obtained when class <EM>c</EM> should have
been found.</P>
'''

confmat_format['detail2,latex'] = r'''
Table~\ref{tab:conf-%s} gives the corresponding class confusion matrix.
Here, the element at row $r$ and column $c$ indicates that class $r$ was
obtained when class $c$ should have been found.
'''

confmat_format['detail2,text'] = r'''
The above table gives the corresponding class confusion matrix.
Here, the element at row r and column c indicates that class r was
obtained when class c should have been found.
'''


# The following formats are used by format_comparison.
comparison_format = {}
comparison_format['detail2,html'] = """
The comparison in the table was performed using McNemar's test:
$$Z = \\frac{(|N_{sf} - N_{fs}| - 1)}{\sqrt{N_{sf} + N_{fs}}}$$
where $N_{sf}$ is the number of cases where the first algorithm succeeded
and the second failed, <EM>etc</EM>.  Each line shows the better-performing
algorithm."""

comparison_format['detail2,latex'] = r"""
The comparison in the table was performed using McNemar's test:
\begin{equation}
  Z = \frac{(|N_{sf} - N_{fs}| - 1)}{\sqrt{N_{sf} + N_{fs}}}
\end{equation}
where $N_{sf}$ is the number of cases where the first algorithm succeeded
and the second failed, \emph{etc}.  Each line shows the better-performing
algorithm."""

comparison_format['detail2,text'] = """
The comparison in the table was performed using McNemar's test:

  Z = |Nsf - Nfs| - 1
      ---------------
         _________
       \|Nsf + Nfs

where Nsf is the number of cases where the first algorithm succeeded
and the second failed, etc.  Each line shows the better-performing
algorithm."""

review_header = 'No Tests  Class'


#-----------------------------------------------------------------------------
# R O U T I N E S
#-----------------------------------------------------------------------------
def analyse (resfile, fmt, detail=2):
    '''Analyse a file of results.'''
    # Load the file and set things up.  Then branch according to the type of
    # tests defined by the script that generated the results.
    content = load_script (resfile, reportnargs, extension='.res')
    type = content['transcript_begin'][0][2]
    if type == 'label':
        classes, rates = error_rates (content['result'])
        print(format_error_rates (classes, rates, fmt, resfile, detail))
        ccm, exp, act = confusion_matrix (content['result'])
        print(format_confusion_matrix (ccm, exp, act, fmt, resfile, detail))
    else:
        print('Unknown experiment type of "' + type + '".', file=sys.stderr)

#-----------------------------------------------------------------------------
def compare (transcripts, fmt, detail=2):
    '''Compare a set of transcripts'''
    # Load the transcript files and ensure they were all generated from the
    # same test script.
    tscripts = []
    s1 = v1 = None
    for file in transcripts:
        t = load_script (file, reportnargs, extension='.res')
        if s1 is None:
            s1 = t['transcript_begin'][0][0]
            v1 = t['transcript_begin'][0][1]
        else:
            s2 = t['transcript_begin'][0][0]
            v2 = t['transcript_begin'][0][1]
            if s1 != s2 or v1 != v2:
                print('Script or version mismatch between', \
                    transcripts[0], 'and', file, file=sys.stderr)
                exit (1)
        tscripts.append (t)

    # Identify the various classes of result.
    c = {}
    for ts in tscripts:
        for t in ts['result']:
            c[t[1]] = 1
            c[t[3]] = 1
    classes = sorted (c.keys())

    # Now cycle through the transcripts, in each case comparing the various
    # tests via McNemar's formula.
    n = len (tscripts[0]['result'])
    results = {}
    for s1 in range(0, len(tscripts)):
        for s2 in range(s1+1, len(tscripts)):
            if s1 == s2: continue
            r1 = tscripts[s1]['result']
            r2 = tscripts[s2]['result']
            assert len (r1) == len (r2)
            for c in classes:
                results[c] = mcnemar (r1, r2, c)
            results['overall'] = mcnemar (r1, r2, None)
            print(format_comparison (transcripts[s1], transcripts[s2],
                                     classes, results, fmt, detail))

#-----------------------------------------------------------------------------
def confusion_matrix (results):
    '''Work out and return the class confusion matrix'''
    ccmdata = {}    # indexed by "expected,actual"
    expnames = {}   # expected class names
    actnames = {}   # actual class names

    # Gather up the results in a dictionary, finding the expected and
    # actual class names as we do it.
    for t in results:
        name, expected, status, actual = t
        expnames[expected] = 1
        actnames[actual] = 1
        k = expected + ',' + actual
        if k not in ccmdata: ccmdata[k] = 0
        ccmdata[k] += 1
    return ccmdata, expnames, actnames

#-----------------------------------------------------------------------------
def error_rates (results):
    '''Calculate the error rates from the results in the transcript'''
    # Identify the various classes of result.
    c = {}
    for t in results:
        c[t[1]] = 1
        c[t[3]] = 1
    classes = sorted (c.keys())

    # Work out and report the number of TPs etc for each class and overall.
    count = {}
    mat = {}
    ttp = ttn = tfp = tfn = tn = 0
    for c in classes:
        n = 0
        count['TP'] = count['FP'] = count['TN'] = count['FN'] = 0
        for t in results:
            if t[1] == c:
                r = outcome (t[1], t[2], t[3])
                count[r] += 1
                n += 1
            a, r, p, s = measures (count['TP'], count['TN'],
                                        count['FP'], count['FN'], n)
            mat[c] = [n, count['TP'], count['TN'], count['FP'], \
                          count['FN'], a, r, p, s]

        ttp += count['TP']
        ttn += count['TN']
        tfp += count['FP']
        tfn += count['FN']
        tn  += n
    a, r, p, s = measures (ttp, ttn, tfp, tfn, tn)
    mat['overall'] = [tn, ttp, ttn, tfp, tfn, a, r, p, s]
    return classes, mat

#-----------------------------------------------------------------------------
def execute (script, iface, printres):
    '''Carry out the tests in script'''
    import datetime, time

    # Load the test script and do any checking of it that we can.
    content = load_script (script, scriptnargs)
    nt = int (content['tests'][0])
    na = len (content['test'])
    if nt != na:
        print('Warning: script identifies', nt, \
            'tests but there are actually %d.' % na, file=sys.stderr)

    # Output the start-of-transcript message and start the run timer.
    if printres:
        print('transcript_begin', content['name'][0], content['version'][0], \
            content['type'][0], datetime.datetime.now ())
        start = time.time ()

    # Do the actual tests and output what happened to the transcript.
    results = []
    for t in content['test']:
        s, a = run_test (iface, t[0], t[1], t[2])
        if s: st = 'S'
        else: st = 'F'
        results.append ([t[0], t[2], st, a])
        if printres: print('result', t[0], t[2], st, a)
    
    # Stop the run timer and output the end-of-transcript message, then
    # return the results we've collected.
    if printres:
        duration = time.time () - start
        print('transcript_end', duration)
    return results

#-----------------------------------------------------------------------------
def format_comparison (t1, t2, classes, results, fmt, detail=2):
    if fmt == 'text':
        text = 'Comparison of ' + t1 + ' and ' + t2 + '\n  Z-score  class' \
            + "    better\n"
        for c in classes:
            better = t2 if results[c] < 0 else t1
            if results[c] == 0.0: better = "neither"
            val = abs (results[c])
            text += ('%9.2f  %-7s  %s\n' % (val, c, better))
    elif fmt == 'latex':
        text = '\\begin{table}\n  \\begin{center}\n    \\begin{tabular}{rcl}\n'
        text += '      \\toprule \n'
        text += '      \\multicolumn{1}{c}{\\textsc{Z-score}} &'
        text += '\\multicolumn{1}{c}{\\textsc{class}} & '
        text += '\\multicolumn{1}{c}{\\textsc{better}} \\\\\n'
        text += '      \\midrule\n'
        for c in classes:
            better = t2 if results[c] < 0 else t1
            if results[c] == 0.0: better = "neither"
            val = abs (results[c])
            text += ('    %7.2f & %s & %s \\\\\n' % (val, c, better))
        text += '      \\bottomrule\n    \\end{tabular}\n  \\end{center}\n'
        text += '  \\caption{Comparison of ' + t1 + ' and ' + t2 + '}\n'
        text += '  \label{tab:comp-' + t1 + t2 + '}\n\end{table}'
    elif fmt == 'html':
        text = '<TABLE ALIGN="center" BORDER="1">\n'
        text += '<TR><TH>Z-score</TH> <TH>class</TH> <TH>better</TH></TR>\n'
        for c in classes:
            better = t2 if results[c] < 0 else t1
            if results[c] == 0.0: better = "neither"
            val = abs (results[c])
            text += ('<TR><TD>%7.2f </TD><TD ALIGN="center"> %s </TD>' \
                     '<TH> %s</TH></TR>\n') % (val, c, better)
        text += '</TABLE>\n'
        text += '<P ALIGN="center">Comparison of ' + t1 + ' and ' + t2 + '</P>\n'
    if detail >= 2:
        text += comparison_format['detail2,' + fmt]
    return text

#-----------------------------------------------------------------------------
def format_confusion_matrix (ccmdata, expnames, actnames, fmt, fn, detail=2):
    '''Format a confusion matrix in one of the supported formats.'''

    # Convert the dictionaries containing the matrix's data into a list of
    #  lists, which we format as a table and return.
    ccm = [['actual']]
    for e in sorted (expnames.keys()):
        ccm[0].append (e)
    for a in sorted (actnames.keys()):
        vals = []
        vals.append (a)
        for e in sorted (expnames.keys()):
            k = e + ',' + a
            if k not in ccmdata: vals.append (0)
            else: vals.append (ccmdata[k])
        ccm.append (vals)

    text = format_table (ccm, fmt=fmt, coltitles=True, rowtitles=True,
                         datafmt='%9d', colfmt='%9s', rowfmt='%9s',
                         rowtitle='actual', coltitle='expected',
                         caption='Confusion matrix calculated from ' + fn,
                         label='tab:conf-' + fn)
    if detail >= 2:
        if fmt == 'latex': text += confmat_format['detail2,latex'] % fn
        else:              text += confmat_format['detail2,' + fmt]
    return text

#-----------------------------------------------------------------------------
def format_error_rates (classes, rates, fmt, fn, detail=2):
    '''Return a table of the various error rates in the relevant format'''
    text = error_rate_format['header,' + fmt] % fn
    for c in classes:
        v = rates[c][:]
        v.append (c)
        if c != 'overall':
            text += error_rate_format['layout,' + fmt] % tuple(v)
    text += error_rate_format['separator,' + fmt]
    v = rates['overall'][:]
    v.append ('overall')
    text += error_rate_format['layout,' + fmt] % tuple(v)
    if fmt == 'text':
        text += error_rate_format['footer,text']
        if detail >= 2: text += error_rate_format['detail2,text'] % fn
    elif fmt == 'latex':
        text += error_rate_format['footer,latex'] % (fn, fn)
        if detail >= 2: text += error_rate_format['detail2,latex'] % (fn, fn, fn)
    elif fmt == 'html':
        text += error_rate_format['footer,html'] % fn
        if detail >= 2: text += error_rate_format['detail2,html'] % fn
    return text

#-----------------------------------------------------------------------------
def format_table (data, fmt='latex', coltitles=False, rowtitles=False,
                  datafmt = '%s', colfmt='%s', rowfmt='%s',
                  rowtitle='', coltitle='', caption='', label=''):
    "Format a table for HTML, LaTeX or text and return the resulting text."
    rows = len (data)
    first_col = 1 if rowtitles else 0
    cols = len (data[0])
    if fmt == 'latex':
        # Generate the top of the table.
        text = '\\begin{table}\n  \\begin{center}\n    \\begin{tabular}{'
        if rowtitles: text += 'c'
        text += 'r' * (cols - first_col)
        text += '}\n      \\toprule\n'
        # Next, do the main body of the table.
        for row_no in range (0, rows):
            text += '      '
            row = data[row_no]
            if row_no == 0 and coltitles:
                if rowtitles:
                    text += '& \\multicolumn{%d}{c}{\\bfseries %s}\\\\' \
                            % (cols - first_col, coltitle)
                    text += '\\cline{2-%d}' % (cols - first_col + 1)
                    text += '\\textbf{%s} & ' % rowtitle
                if coltitles:
                    t = 'c'
                    for col in range (first_col, cols-1):
                        text += '\\multicolumn{1}{' + t + '}{\\bfseries '
                        text += colfmt % row[col] + '} & '
                        t = 'c'
                    text += '\\multicolumn{1}{' + t + '}{\\bfseries '
                    text += colfmt % row[cols-1]
                    text += '}\\\\\n      \\midrule\n'
            else:
                if rowtitles:
                    text += '\\multicolumn{1}{c}{\\bfseries '
                    text += rowfmt % row[0]
                    text += '} & '
                for col in range (first_col, cols-1):
                    text += datafmt % row[col] + ' & '
                text += datafmt % row[cols-1]
                text += '\\\\\n'
        # Finish off the table.
        text += '       \\bottomrule\n'
        text += '    \\end{tabular}\n  \end{center}\n'
        if caption != '': text += '  \\caption{' + caption + '}\n'
        if label != '': text += '  \\label{' + label + '}\n'
        text += '\\end{table}\n'
    elif fmt == 'html':
        # Generate the top of the table.
        text = '<TABLE ALIGN="center" BORDER="1">\n'
        # Next, do the main body of the table.
        for row_no in range (0, rows):
            text += '<TR>\n'
            row = data[row_no]
            if row_no == 0 and coltitles:
                if rowtitles:
                    text += ' <TD></TD> <TD COLSPAN="%d" ALIGN="center">' \
                            % (cols - first_col + 1)
                    text += '%s </TD></TR>\n<TR>' %  coltitle
                    text += '  <TD> ' + rowtitle + '</TD>\n'
                if coltitles:
                    for col in range (first_col, cols):
                        text += '  <TH>'
                        text += colfmt % row[col] + '</TH>\n'
                    text += '</TR>\n'
            else:
                if rowtitles:
                    text += '  <TH>'
                    text += rowfmt % row[0]
                    text += '</TH>\n'
                for col in range (first_col, cols):
                    text += '  <TD ALIGN="right"> ' + datafmt % row[col] \
                            + ' </TD>\n'
                text += '</TR>\n'
        # Finish off the table.
        text += '</TABLE>\n</P>\n'
        if caption != '': text += '<P ALIGN="center">' + caption + '</P>\n'
    elif fmt == 'text':
        text = ''
        if caption != '': text += caption + '\n'
        if rowtitles:
            word = colfmt % ' '
            ns = (cols * len(word) - len(coltitle)) // 2 + len(rowtitle)
            text += ' ' * ns + coltitle + '\n'
        for row_no in range (0, rows):
            row = data[row_no]
            if row_no == 0 and coltitles:
                if rowtitles: text += rowfmt % rowtitle
                if coltitles:
                    for col in range (first_col, cols):
                        text += colfmt % row[col]
                    text += '\n'
            else:
                if rowtitles:
                    text += rowfmt % row[0]
                for col in range (first_col, cols):
                    text += datafmt % row[col]
                text += '\n'

    return text

#-----------------------------------------------------------------------------
def help ():
    """Print out the program's help string and exit"""
    print(__doc__, file=sys.stderr)
    exit (1)

#-----------------------------------------------------------------------------
def list_to_string (l, delim=' '):
    "Convert a list of words to a string, with each word separated by delim."
    text = ''
    for i in range (0, len(l)-1):
        e = l[i]
        text += e
        text += delim
    text += l[-1]
    return text

#-----------------------------------------------------------------------------
def load_script (script, nargs, extension='.fact'):
    """
    Load the contents of a file 'script' that conforms to the FACT syntax.

    We return a dictionary, indexed by 'verb', whose content is a list; there
    is an element in that list for each time 'verb' appeared in the file, and
    each element will itself be a list if the 'verb' takes several arguments.
    """
    # Put the default extension in place if there isn't one.
    root, ext = os.path.splitext (script)
    if ext == '': name = script + extension
    else: name = script

    # We read the script's content differently if it's a file or a URL.
    if name[0:7] == 'http://':               # It's a URL
        import urllib.request, urllib.error, urllib.parse, string
        f = urllib.request.urlopen(name)
        lines = string.split (f.read(), '\n')
    else:                                      # It's a file
        f = open (name, 'r')
        lines = f.read().split('\n')
    f.close ()
    return parse_file (lines, nargs)

#-----------------------------------------------------------------------------
def load_interface (interface):
    '''Load an interface module'''
    # Check and sanitize the name of the interface module, then load it.
    if interface.endswith ('.py'): interface = interface[:-3]
    return __import__(interface, globals(), locals(), [''])

#-----------------------------------------------------------------------------
def mcnemar (r1, r2, c):
    """Compare two sets of results using McNemar's test, returning the Z-score
    and an indication of which was better."""
    n = len (r1)
    if len (r2) != n:
        print('Warning: comparing results of different lengths!', file=sys.stderr)
    Nss = Nff = Nfs = Nsf = 0
    for i in range (0, n):
        if c is None or r1[i][1] == c or r2[i][1] == c:
            s1 = sf (r1[i][1], r1[i][3])
            s2 = sf (r2[i][1], r2[i][3])
            if   s1 and s2:     Nss += 1
            elif s1 and not s2: Nsf += 1
            elif not s1 and s2: Nfs += 1
            else:               Nff += 1

    if Nsf + Nfs <= 0:
        z = 0.0
    else:
        z = math.sqrt ((abs(Nsf - Nfs) - 1.0)**2 / (Nsf + Nfs))
    if Nsf == Nfs:
        better = 0
    elif Nsf > Nfs:
        better = 1
    else:
        better = -1
    return z * better

#-----------------------------------------------------------------------------
def measures (tp, tn, fp, fn, n):
    '''Calculate and return accuracy, recall, precision and specificity
    from TP etc'''
    if n == 0: a = 0.0
    else: a = (tp + tn + 0.0) / n
    if tp == 0 and fn == 0: r = 0.0
    else: r = tp / (tp + fn + 0.0)
    if tp == 0 and fp == 0: p = 0.0
    else: p = tp / (tp + fp + 0.0)
    if tn == 0 and fp == 0: s = 0.0
    else: s = tn / (tn + fp + 0.0)
    return a, r, p, s

#-----------------------------------------------------------------------------
def outcome (expected, status, actual):
    '''Determine whether a test resulted in a TP etc'''
    if status == 'S':
        if actual == expected: result = 'TP'
        else:                  result = 'FP'
    elif status == 'F':
        if expected == 'F':    result = 'TN'
        else:                  result = 'FN'
    return result

#-----------------------------------------------------------------------------
def parse_file (lines, nargs):
    """
    Parse the contents of a list of lines that conforms to the FACT syntax.
    Each line starts with a 'verb' which contains a number of arguments; the
    number is given in the dictionary argument 'nargs'.  Lines may be
    continued onto further physical lines by making the last character of each
    line but the final one a backslash character.  (Most of the complexity in
    the routine is due to the support for continuation lines.)  Blank lines
    and lines whose first non-whitespace character is a hash are ignored.

    We return a dictionary, indexed by 'verb', whose content is a list; there
    is an element in that list for each time 'verb' appeared in the file, and
    each element will itself be a list if the 'verb' takes several arguments.
    """
    content = {}
    delim = None
    append = False
    verb = ''
    for line in lines:
        # Split the line into words and append them to argument for the
        # current verb if this is a continuation line.  Otherwise, ignore it
        # if blank or a comment, or split off the verb.
        words = line.split (delim, 1)
        if append:
            rest += ' '.join (words)
            append = False
        else:
            if len (words) == 0: continue
            verb = words[0]
            if verb[0] == '#': continue
            rest = words[1:]
        # Determine whether this line will be continued.
        if len (rest) < 1:
            lastchar = ''
        else:
            rest = ''.join (rest)
            lastchar = rest[-1]
        if lastchar == '\\':
            append = True
            rest = rest[:-1]
        else:
            # We are finally able to store this line (possibly after it has
            # been continued).  Split it into the right number of words, make
            # sure the entry in the dictionary is a list-head, and append the
            # arguments to the list.
            if nargs[verb] > 1: rest = rest.split (delim, nargs[verb]-1)
            if verb in content:
                content[verb].append (rest)
            else:
                content[verb] = []
                content[verb].append (rest)
    return content

#-----------------------------------------------------------------------------
def review (script):
    '''Review the tests in a test script'''
    text = ''
    fmt = '%7s:  %s\n'
    # Load the test script and do any checking of it that we can.
    content = load_script (script, scriptnargs)
    nt = int (content['tests'][0])
    na = len (content['test'])
    if nt != na:
        print('Warning: script identifies', nt, \
            'tests but there are actually %d.' % na, file=sys.stderr)

    # Summarize the information stored in the test script.
    text += (fmt % ('Name', content['name'][0]))
    text += (fmt % ('URL', content['url'][0]))
    text += (fmt % ('Author', list_to_string(content['author'][0])))
    text += (fmt % ('Type', content['type'][0]))
    text += (fmt % ('Version', content['version'][0]))
    text += (fmt % ('Purpose', content['purpose'][0]))

    # Walk through the tests and see how many times each class is tested for.
    classes = {}
    classes['failure'] = 0
    for t in content['test']:
        k = t[2]
        if k not in classes: classes[k] = 1
        else:  classes[k] += 1

    # Produce a table of the class occupancy.
    ckeys = sorted (classes.keys())
    ckeys.append ('Total')
    classes['Total'] = nt
    text += review_header + '\n'
    for c in ckeys:
        text += ("%8d  %s\n" % (classes[c], c))
    return text[:-1]

#-----------------------------------------------------------------------------
def run_test (interface, name, input, expected):
    '''Run a single test and determine whether it yielded a TP etc'''
    return interface.interface (name, input)

#-----------------------------------------------------------------------------
def sf (e, a):
    if a == 'F':
        s = False
    elif e == a:
        s = True
    else:
        s = False
    return s

#-----------------------------------------------------------------------------
def valof (symbol, default):
    '''Return the value of 'symbol' from our symbol table'''
    if symbol not in symtab:
        return default
    return symtab[symbol]

#-----------------------------------------------------------------------------
def plot (x, y, title, xlabel, ylabel, logx=False, logy=False):
    '''Plot a graph using Gnuplot'''
    p = os.popen ('gnuplot', 'w')
    print('set nokey', file=p)
    print('set grid', file=p)
    print('set title "' + title + '"', file=p)
    print('set xlabel "' + xlabel + '"', file=p)
    print('set ylabel "' + ylabel + '"', file=p)
    print('set style data linespoints', file=p)
    if logx: print('set log x')
    if logy: print('set log y')
    print('plot "-"', file=p)
    for ix, iy in zip(x, y):
        print(ix, iy, file=p)
    print('e', file=p)
    p.flush ()
    # Exit if the user types <EOF>; give (minimal) instructions if they type
    # "?"; simply continue if they type <return>.  Anything else typed in
    # response to the prompt is assumed to be a filename and we save the data
    # that file. (And yes, that can result in silly filenames...)
    looping = True
    while looping:
        sys.stderr.write ('CR> ')
        fn = sys.stdin.readline()
        if len(fn) < 1:
            print("Exiting...", file=sys.stderr)
            sys.exit (1)
        if len(fn) > 0 and fn == '?\n':
            print('fn to save data to "fn" else <return>', file=sys.stderr)
            continue
        if len(fn) > 1 and fn != '':
            f = open (fn[:-1],'w')
            for ix, iy in zip(x, y):
                print(ix, iy, file=f)
            f.close ()
        looping = False
    p.close ()

#-----------------------------------------------------------------------------
# M A I N    P R O G R A M
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # Parse the command line.
    parser = optparse.OptionParser(description='''FACT is the Framework for
Algorithm Comparison and Testing, a test harness which you can use for
evaluating the performance of statistically-based techniques, especially in
the context of computer vision.''')
    parser.add_option ('-V', '--version', dest='version', action='store_true',
                       default=False, help='output program version')
    parser.add_option ('-H', '--head', dest='head', action='store_true',
                       default=False, help='generate preamble for format')
    parser.add_option ('-T', '--tail', dest='tail', action='store_true',
                       default=False, help='generate postamble for format')
    parser.add_option ('-d', '--detail',  dest='detail', type='int',
                       default=1, help='set level of detail in output')
    parser.add_option ('-f', '--format',  dest='format', type='choice',
                       choices=('html', 'latex', 'text'), default='text',
                       help='output format')
    parser.add_option ('-i', '--interface', dest='interface',
                       default='interface', help='name of interface module')
    (options, args) = parser.parse_args()

    # Ensure everything is defined.
    if options.version:
        print('FACT version', timestamp[13:-1])
        exit (1)
    symtab = {}

    # Generate the preamble for the chosen format, if required.
    if options.head: print(preamble[options.format])

    # Work out what we are to do and ensure we have at least the task
    # we're to do on the command line.  If we do, branch to the
    # appropriate section of code.
    nargs = len (args)
    if nargs < 1: help ()
    task = args[0]

    if task == 'analyse' or task == 'analyze' or task == 'anal':
        if nargs != 2: help ()
        transcript = args[1]
        analyse (args[1], options.format, options.detail)

    elif task == 'compare' or task == 'comp':
        if nargs < 3: help ()
        compare (args[1:], options.format, options.detail)

    elif task == 'execute' or task == 'run':
        if nargs != 2: help ()
        iface = load_interface (options.interface)
        execute (args[1], iface, True)

    elif task == 'help':
        help ()

    elif task == 'review':
        if nargs < 2: help ()
        print(review (args[1]))

    elif task == 'optimise' or task == 'optimize' or task == 'opt':
        if nargs < 3: help ()
        # Process the tuning parameters.
        opts = args[1:-1]
        for opt in opts:
            param,valstring = opt.split('=', 1)
            vals = valstring.split(',')
        iface = load_interface (options.interface)
        print('Processing script', args[-1])

    elif task == 'vary' or task == 'roc':
        if nargs < 3: help ()
        # Process the tuning parameters.
        param,valstring = args[1].split('=', 1)
        vals = valstring.split(',')
        # Load the interface file and carry out the various runs,
        # saving the results for later.
        iface = load_interface (options.interface)
        variation = {}
        for v in vals:
            symtab[param] = v
            results = execute (args[2], iface, False)
            classes, rates = error_rates (results)
            variation[v] = rates['overall'][:]

        # Finally, generate the output.
        x = []; y = []; pre = []; rec = []; sens = []; spec = []
        for v in vals:
            x.append (variation[v][3])
            y.append (variation[v][1])
            pre.append (variation[v][7])
            rec.append (variation[v][6])
            sens.append (variation[v][6])   # same as recall
            spec.append (variation[v][8])
        plot (x, y, 'ROC Curve', 'false positives', 'true positives')
        plot (pre, rec, 'Precision-Recall Curve', 'precision', 'recall')
        plot (sens, spec, 'Sensitivity-Specificity Curve',\
                  'sensitivity', 'specificity')

    else:
        print('Invalid task "' + task + '"', file=sys.stderr)
        help ()

    # Generate the postamble for the chosen format, if required.
    if options.head: print(postamble[options.format])

# Local Variables:
# time-stamp-line-limit: 100
# End:
#-----------------------------------------------------------------------------
# End of fact.
#-----------------------------------------------------------------------------
