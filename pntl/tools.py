
# encoding: utf-8
# Practical Natural Language Processing Tools (practNLPTools-lite):
#               Combination of Senna and Stanford dependency Extractor
# Copyright (C) 2017 PractNLP-lite Project
# Current Author: Jawahar S <jawahar273@gmail.com>
# URL: https://jawahar273.gitbooks.io (or) https://github.com/jawahar273


from __future__ import generators, print_function, unicode_literals
import subprocess
import json
import tree_kernel
import subprocess
import os
import re
import spacy
from platform import architecture, system
from allennlp.predictors import Predictor
from allennlp.service.predictors import SemanticRoleLabelerPredictor
#from pattern.en import conjugate, lemma, lexeme,PRESENT,SG

try:
    from colorama import init
    from colorama.Fore import RED, BLUE
    init(autoreset=True)
except ImportError:
    RED = " "
    BLUE = " "


class Annotator:
    """
    :param str senna_dir: path where is located
    :param str dep_model: Stanford dependencie mode
    :param str stp_dir: path of stanford parser jar
    :param str raise_e: raise exception if stanford-parser.jar
                        is not found
    """

    def __init__(self, senna_dir='', stp_dir='',
                 dep_model='edu.stanford.nlp.trees.'
                           'EnglishGrammaticalStructure',
                 raise_e=False):
        """
        init function of Annotator class
        """
        self.senna_path = ''
        self.dep_par_path = ''

        if not senna_dir:
            if 'SENNA' in os.environ:
                self.senna_path = os.path.normpath(os.environ['SENNA'])
                self.senna_path + os.path.sep
                exe_file_2 = self.get_senna_bin(self.senna_path)
                if not os.path.isfile(exe_file_2):
                    raise OSError(RED +
                                  "Senna executable expected at %s or"
                                  " %s but not found"
                                  % (self.senna_path, exe_file_2))
        elif senna_dir.startswith('.'):
            self.senna_path = os.path.realpath(senna_dir) + os.path.sep
        else:
            self.senna_path = senna_dir.strip()
            self.senna_path = self.senna_path.rstrip(os.path.sep) + os.path.sep

        if not stp_dir:
            import pntl.tools as Tfile
            self.dep_par_path = Tfile.__file__.rsplit(os.path.sep, 1)[0] + os.path.sep
            self.check_stp_jar(self.dep_par_path, raise_e=True)
        else:
            self.dep_par_path = stp_dir + os.path.sep
            self.check_stp_jar(self.dep_par_path, raise_e)

        self.dep_par_model = dep_model
        # print(dep_model)

        self.default_jar_cli = ['java', '-cp', 'stanford-parser.jar',
                                self.dep_par_model,
                                '-treeFile', 'in.parse', '-collapsed']
        self.print_values()

    def print_values(self):
        """ displays the current set of values
        such as SENNA location, stanford parser jar,
        jar command interface
        """
        print("**" * 50)
        print("default values:\nsenna path:\n", self.senna_path,
              "\nDependencie parser:\n", self.dep_par_path)
        # print(self.default_jar_cli)
        print("Stanford parser clr", " ".join(self.default_jar_cli))
        print("**" * 50)

    def check_stp_jar(self, path, raise_e=False, _rec=True):
        """Check the stanford parser is present in the given directions
        and nested searching will be added in futurwork

        :param str path: path of where the stanford parser is present
        :param bool raise_e: to raise exception with user
              wise and default `False` don't raises exception
        :return: given path if it is valid one or return boolean `False` or
             if raise FileNotFoundError on raise_exp=True
        :rtype: bool

        """
        gpath = path
        path = os.listdir(path)
        file_found = False
        for file in path:
            if file.endswith(".jar"):
                if file.startswith("stanford-parser"):
                    file_found = True
        if not file_found:
            # need to check the install dir for stanfor parser
            if _rec:
                import pntl
                path_ = os.path.split(pntl.__file__)[0]
                self.check_stp_jar(path_, raise_e, _rec=False)
            if raise_e:
                raise FileNotFoundError(RED + "`stanford-parser.jar` is "
                                        "not"
                                        " found in the path \n"
                                        "`{}` \n"
                                        "To know about more about the issues,"
                                        "got to this given link ["
                                        "http://pntl.readthedocs.io/en/"
                                        "latest/stanford_installing_"
                                        "issues.html] \n User "
                                        "`pntl -I true` to downlard "
                                        "needed file automatically."
                                        .format(gpath))
        return file_found

    @property
    def stp_dir(self):
        """The return the path of stanford parser jar location
        and set the path for Dependency Parse at run time(
        this is python @property)
        """
        return self.dep_par_path

    @stp_dir.setter
    def stp_dir(self, val):
        if os.path.isdir(val):
            self.dep_par_path = val + os.path.sep

    @property
    def senna_dir(self):
        """The return the path of senna location
        and set the path for senna at run time(this is python @property)

        :rtype: string
        """
        return self.senna_path

    @senna_dir.setter
    def senna_dir(self, val):
        if os.path.isdir(val):
            self.senna_path = val + os.path.sep

    @property
    def jar_cli(self):
        """
        The return cli for standford-parser.jar(this is python @property)

        :rtype: string
        """
        return " ".join(self.default_jar_cli)

    @jar_cli.setter
    def jar_cli(self, val):
        self.default_jar_cli = val.split()

    def get_senna_bin(self, os_name):
        """
        get the current os executable binary file.

        :param str os_name: os name like Linux, Darwin, Windows
        :return: the corresponding exceutable object file of senna
        :rtype: str
        """

        if os_name == 'Linux':
            bits = architecture()[0]
            if bits == '64bit':
                executable = 'senna-linux64'
            elif bits == '32bit':
                executable = 'senna-linux32'
            else:
                executable = 'senna'
        elif os_name == 'Darwin':
            executable = 'senna-osx'
        elif os_name == 'Windows':
            executable = 'senna-win32.exe'
        return self.senna_path + executable

    def get_senna_tag_batch(self, sentences):
        """
        Communicates with senna through lower level communiction(sub process)
        and converted the console output(default is file writing).
        On batch processing each end is add with new line.

        :param list sentences: list of sentences for batch processes
        :rtype: str
        """
        input_data = ""
        for sentence in sentences:
            input_data += sentence + "\n"
        input_data = input_data[:-1]
        package_directory = os.path.dirname(self.senna_path)
        os_name = system()
        executable = self.get_senna_bin(os_name)
        senna_executable = os.path.join(package_directory, executable)
        cwd = os.getcwd()
        os.chdir(package_directory)
        pipe = subprocess.Popen(senna_executable,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=True)
        senna_stdout = pipe.communicate(input=input_data.encode('utf-8'))[0]
        os.chdir(cwd)
        return senna_stdout.decode().split("\n\n")[0:-1]

    @classmethod
    def help_conll_format(cls):
        """With the help of this method, detail of senna
         arguments are displayed
        """
        return cls.get_conll_format.__doc__.split("\n\n")[1]

    def get_conll_format(self, sentence, options='-srl -pos -ner -chk -psg'):
        """Communicates with senna through lower level communiction
        (sub process) and converted the console output(default is file writing)
        with CoNLL format and argument to be in `options` pass

        :param str or list: list of sentences for batch processes
        :param  options list: list of arguments
+--------------+-----------------------------------------------+
| options      | desc                                          |
+==============+===============================================+
| -verbose     | Display model informations (on the standard   |
|              | error output, so it does not mess up the tag  |
|              | outputs).                                     |
+--------------+-----------------------------------------------+
| -notokentags | Do not output tokens (first output column).   |
+--------------+-----------------------------------------------+
| -offsettags  | Output start/end character offset (in the     |
|              | sentence), for each token.                    |
+--------------+-----------------------------------------------+
| -iobtags     | Output IOB tags instead of IOBES.             |
+--------------+-----------------------------------------------+
| -brackettags | Output ‘bracket’ tags instead of IOBES.       |
+--------------+-----------------------------------------------+
| -path        | Specify the path to the SENNA data and hash   |
|              | directories, if you do not run SENNA in its   |
|              | original directory. The path must end by “/”. |
+--------------+-----------------------------------------------+
| -usrtokens   | Use user’s tokens (space separated) instead   |
|              | of SENNA tokenizer.                           |
+--------------+-----------------------------------------------+
| -posvbs      | Use verbs outputed by the POS tagger instead  |
|              | of SRL style verbs for SRL task. You might    |
|              | want to use this, as the SRL training task    |
|              | ignore some verbs (many “be” and “have”)      |
|              | which might be not what you want.             |
+--------------+-----------------------------------------------+
| -usrvbs      | Use user’s verbs (given in ) instead of SENNA |
|              | verbs for SRL task. The file must contain one |
|              | line per token, with an empty line between    |
|              | each sentence. A line which is not a “-”      |
|              | corresponds to a verb.                        |
+--------------+-----------------------------------------------+
| -pos         | Instead of outputing tags for all tasks,      |
|              | SENNA will output tags for the specified (one |
|              | or more) tasks.                               |
+--------------+-----------------------------------------------+
| -chk         | Instead of outputing tags for all tasks,      |
|              | SENNA will output tags for the specified (one |
|              | or more) tasks.                               |
+--------------+-----------------------------------------------+
| -ner         | Instead of outputing tags for all tasks,      |
|              | SENNA will output tags for the specified (one |
|              | or more) tasks.                               |
+--------------+-----------------------------------------------+
| -srl         | Instead of outputing tags for all tasks,      |
|              | SENNA will output tags for the specified (one |
|              | or more) tasks.                               |
+--------------+-----------------------------------------------+
| -psg         | Instead of outputing tags for all tasks,      |
|              | SENNA will output tags for the specified (one |
|              | or more) tasks.                               |
+--------------+-----------------------------------------------+

        :return: senna tagged output
        :rtype: str
        """
        if isinstance(options, str):
            options = options.strip().split()

        input_data = sentence
        package_directory = os.path.dirname(self.senna_path)
        os_name = system()
        executable = self.get_senna_bin(os_name)
        senna_executable = os.path.join(executable)
        # print("testing dir", executable, package_directory)
        cwd = os.getcwd()
        os.chdir(package_directory)
        args = [senna_executable]
        args.extend(options)
        pipe = subprocess.Popen(args,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=True)
        senna_stdout = pipe.communicate(input=" ".join(input_data)
                                        .encode('utf-8'))[0]
        os.chdir(cwd)
        return senna_stdout.decode("utf-8").strip()

    def get_senna_tag(self, sentence):
        """
        Communicates with senna through lower level communiction(sub process)
        and converted the console output(default is file writing)

        :param str/list listsentences : list of sentences for batch processes
        :return: senna tagged output
        :rtype: str
        """
        input_data = sentence
        package_directory = os.path.dirname(self.senna_path)
        # print("testing dir",self.dep_par_path, package_directory)
        os_name = system()
        executable = self.get_senna_bin(os_name)
        senna_executable = os.path.join(package_directory, executable)
        cwd = os.getcwd()
        os.chdir(package_directory)
        pipe = subprocess.Popen(senna_executable,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                shell=True)
        senna_stdout = pipe.communicate(input=" ".join(input_data)
                                        .encode('utf-8'))[0]
        os.chdir(cwd)
        return senna_stdout

    def get_dependency(self, parse):
        """
        Change to the Stanford parser direction and process the works

        :param str parse: parse is the input(tree format)
                  and it is writen in as file

        :return: stanford dependency universal format
        :rtype: str
        """
        # print("\nrunning.........")
        package_directory = os.path.dirname(self.dep_par_path)
        cwd = os.getcwd()
        os.chdir(package_directory)

        with open(self.senna_path + os.path.sep + "in.parse",
                  "w", encoding='utf-8') as parsefile:
            parsefile.write(parse)
        pipe = subprocess.Popen(self.default_jar_cli,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True)
        pipe.wait()

        stanford_out = pipe.stdout.read()
        # print(stanford_out, "\n", self.default_jar_cli)
        os.chdir(cwd)

        return stanford_out.decode("utf-8").strip()

    def get_batch_annotations(self, sentences, dep_parse=True):
        """
        :param list sentences: list of sentences
        :rtype: dict
        """
        annotations = []
        batch_senna_tags = self.get_senna_tag_batch(sentences)
        for senna_tags in batch_senna_tags:
            annotations += [self.get_annoations(senna_tags=senna_tags)]
        if dep_parse:
            syntax_tree = ""
            for annotation in annotations:
                syntax_tree += annotation['syntax_tree']
            dependencies = self.get_dependency(syntax_tree).split("\n\n")
            # print (dependencies)
            if len(annotations) == len(dependencies):
                for dependencie, annotation in zip(dependencies, annotations):
                    annotation["dep_parse"] = dependencie
        return annotations

    def get_annoations(self, sentence='', senna_tags=None, dep_parse=True):
        """
        passing the string to senna and performing aboue given nlp process
        and the returning them in a form of `dict()`

        :param str or list sentence: a sentence or list of
                     sentence for nlp process.
        :param str or list senna_tags:  this values are by
                     SENNA processed string
        :param bool  batch: the change the mode into batch
                     processing process
        :param bool dep_parse: to tell the code and user need
                    to communicate with stanford parser
        :return: the dict() of every out in the process
                    such as ner, dep_parse, srl, verbs etc.
        :rtype: dict
        """
        annotations = {}
        if not senna_tags:
            senna_tags = self.get_senna_tag(sentence).decode()
            senna_tags = [x.strip() for x in senna_tags.split("\n")]
            senna_tags = senna_tags[0:-2]
        else:
            senna_tags = [x.strip() for x in senna_tags.split("\n")]
        no_verbs = len(senna_tags[0].split("\t")) - 6

        words = []
        pos = []
        chunk = []
        ner = []
        verb = []
        srls = []
        syn = []
        for senna_tag in senna_tags:
            senna_tag = senna_tag.split("\t")
            words += [senna_tag[0].strip()]
            pos += [senna_tag[1].strip()]
            chunk += [senna_tag[2].strip()]
            ner += [senna_tag[3].strip()]
            verb += [senna_tag[4].strip()]
            srl = []
            for i in range(5, 5 + no_verbs):
                srl += [senna_tag[i].strip()]
            srls += [tuple(srl)]
            syn += [senna_tag[-1]]
        roles = []
        for j in range(no_verbs):
            role = {}
            i = 0
            temp = ""
            curr_labels = [x[j] for x in srls]
            for curr_label in curr_labels:
                splits = curr_label.split("-")
                if splits[0] == "S":
                    if len(splits) == 2:
                        if splits[1] == "V":
                            role[splits[1]] = words[i]
                        else:
                            if splits[1] in role:
                                role[splits[1]] += " " + words[i]
                            else:
                                role[splits[1]] = words[i]
                    elif len(splits) == 3:
                        if splits[1] + "-" + splits[2] in role:
                            role[splits[1] + "-" + splits[2]] += " " + words[i]
                        else:
                            role[splits[1] + "-" + splits[2]] = words[i]
                elif splits[0] == "B":
                    temp = temp + " " + words[i]
                elif splits[0] == "I":
                    temp = temp + " " + words[i]
                elif splits[0] == "E":
                    temp = temp + " " + words[i]
                    if len(splits) == 2:
                        if splits[1] == "V":
                            role[splits[1]] = temp.strip()
                        else:
                            if splits[1] in role:
                                role[splits[1]] += " " + temp
                                role[splits[1]] = role[splits[1]].strip()
                            else:
                                role[splits[1]] = temp.strip()
                    elif len(splits) == 3:
                        if splits[1] + "-" + splits[2] in role:
                            role[splits[1] + "-" + splits[2]] += " " + temp
                            role[splits[1] + "-" + splits[2]] = role[splits[1] + "-" + splits[2]].strip()
                        else:
                            role[splits[1] + "-" + splits[2]] = temp.strip()
                    temp = ""
                i += 1
            if "V" in role:
                roles += [role]
        annotations['words'] = words
        annotations['pos'] = list(zip(words, pos))
        annotations['ner'] = list(zip(words, ner))
        annotations['srl'] = roles
        annotations['verbs'] = [x for x in verb if x != "-"]
        annotations['chunk'] = list(zip(words, chunk))
        annotations['dep_parse'] = ""
        annotations['syntax_tree'] = ""
        for (word_, syn_, pos_) in zip(words, syn, pos):
            annotations['syntax_tree'] += syn_.replace("*", "(" + pos_ + " " + word_ + ")")
        #annotations['syntax_tree']=annotations['syntax_tree'].replace("S1","S")
        if dep_parse:
            annotations['dep_parse'] = self.get_dependency(annotations
                                                           ['syntax_tree'])
        return annotations


def test(senna_path='', sent='',
         dep_model='',
         batch=False,
         stp_dir=''):
    """please replace the path of yours environment(according to OS path)

    .. warning::
       deprecated:: 0.2.0.
       See CLI doc instead. This `test()` function will be removed from next release.

    :param str senna_path: path for senna location \n
    :param str dep_model: stanford dependency parser model location \n
    :param str or list sent: the sentence to process with Senna \n
    :param bool batch:  processing more than one sentence
       in one row \n
    :param str stp_dir: location of stanford-parser.jar file

    """
    from pntl.utils import skipgrams
    annotator = Annotator(senna_path, stp_dir, dep_model)
    if not sent and batch:
        sent = ["He killed the man with a knife and murdered"
                "him with a dagger.",
                "He is a good boy.",
                "He created the robot and broke it after making it."]
    elif not sent:
        sent = 'get me a hotel on chennai in 21-4-2017 '
        # "He created the robot and broke it after making it.
    if not batch:
        print("\n", sent, "\n")
        sent = sent.split()
        args = '-srl -pos'.strip().split()
        print("conll:\n", annotator.get_conll_format(sent, args))
        temp = annotator.get_annoations(sent, dep_parse=True)['dep_parse']
        print('dep_parse:\n', temp)
        temp = annotator.get_annoations(sent, dep_parse=True)['chunk']
        print('chunk:\n', temp)
        temp = annotator.get_annoations(sent, dep_parse=True)['pos']
        print('pos:\n', temp)
        temp = annotator.get_annoations(sent, dep_parse=True)['ner']
        print('ner:\n', temp)
        temp = annotator.get_annoations(sent, dep_parse=True)['srl']
        print('srl:\n', temp)
        temp = annotator.get_annoations(sent,
                                        dep_parse=True)['syntax_tree']
        print('syntaxTree:\n', temp)
        temp = annotator.get_annoations(sent, dep_parse=True)['words']
        print('words:\n', temp)
        print('skip gram\n', list(skipgrams(sent, n=3, k=2)))

    else:
        print("\n\nrunning batch process", "\n", "=" * 20,
              "\n", sent, "\n")
        args = '-srl -pos'.strip().split()
        print("conll:\n", annotator.get_conll_format(sent, args))
        print(BLUE + "CoNLL format is recommented for batch process")
def findDependencyWord( strParam, orderNo ):
    if orderNo == 0:
        prm  = re.compile('\((.*?)-', re.DOTALL |  re.IGNORECASE).findall(strParam)
    elif orderNo == 1:
        prm  = re.compile(', (.*?)-', re.DOTALL |  re.IGNORECASE).findall(strParam)
    if prm :
        return prm[0]

def keyCheck(key, arr, default):
    if key in arr.keys():
        return arr[key]
    else:
        return default

def checkForAppropriateObjOrSub(srls,j,sType):
    if (sType == 0):
        for i in range(0,5):
            if (keyCheck('ARG' + str(i), srls[j], "") != ''):
                return srls[j]['ARG' + str(i)]
    elif (sType == 1):
        foundIndex = 0
        for i in range(0,5):
            if (keyCheck('ARG' + str(i), srls[j], "") != ''):
                foundIndex = foundIndex + 1
                if (foundIndex == 2):
                    return srls[j]['ARG' + str(i)]
    elif (sType == 2):
        foundIndex = 0
        for i in range(0,6):
            if (keyCheck('ARG' + str(i), srls[j], "") != ''):
                foundIndex = foundIndex + 1
                if (foundIndex == 3):
                    return srls[j]['ARG' + str(i)]

    return ''

def getBaseFormOfVerb (verb):
    #return lemma(verb)
    #todo: pattern no longer working use another library!
    with open('verbForms.txt', 'r') as myfile:
        verb = verb.lower()
        f=myfile.read()
        oldString = find_between(f, "", "| "+verb+" ")
        oldString = oldString + '|'
        k = oldString.rfind("<")
        newString = oldString[:k] + "_" + oldString[k+1:]
        if find_between(newString, "_ ", " |") == '':
            if find_between(f, "< "+verb+" "," >") == '':
                with open('logs.txt', "a") as logFile:
                    logFile.write('Warning! Base form of verb '+verb+ ' not found\n')
                    print ('Warning! Base form of verb '+verb+ ' not found')
            return verb

        return find_between(newString, "_ ", " |")

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def getObjectPronun(text):
    with open('objectPronouns.txt', 'r') as myfile:
        f=myfile.read()
        string = find_between(f, '< '+text.lower()+' | ', ' >')
        if(string != ''): return string
        return text

contractions_dict = {
    "ain't": "am not; are not; is not; has not; have not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

def generate(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = ' '.join(text.split())
    text = text.replace(' ,', ',')

    p = re.compile(r'(?:(?<!\w)\'((?:.|\n)+?\'?)(?:(?<!s)\'(?!\w)|(?<=s)\'(?!([^\']|\w\'\w)+\'(?!\w))))')
    subst = "\"\g<1>\""
    text = re.sub(p, subst, text)

    """
    p1 = re.compile(r'(?:(?<!\w)\'((?:.|\n)+?\'?)(?:(?<!s)\'(?!\w)|(?<=s)\'(?!([^\']|\w\'\w)+\'(?!\w))))')
    test_str = "'The Joneses' car'. Similar to the 'Kumbh melas', celebrated by the banks of the holy rivers of India. 'Hey'! you'll, I'm, he's, \nsome random text 'what if it's\nmultiline?'. 'I'm one of the persons''\n\n'the classes' hours'\n\n'the Joneses' car' guys' night out\n'two actresses' \nroles' and 'the hours' of classes'\n\n'The Joneses' car won't start'"
    subst1 = "\"\g<1>\""
    result = re.sub(p1, subst1, test_str)
    print(result)
    """
    predicates = []
    subjects = []
    objects = []
    extraFields = []
    types = []
    #print annotator.getAnnotations("Similarly, as elevation increases there is less overlying atmospheric mass, so that pressure decreases with increasing elevation.",dep_parse=True)

    #text = "REM sleep is characterized by darting movement of closed eyes."
    #text = "In 1996, the trust employed over 7,000 staff and managed another six sites in Leeds and the surrounding area."
    #text = "Early in the twentieth century, Thorstein Veblen, an American institutional economist, analysed cultural influences on consumption."
    #text = "Some of Britain's most dramatic scenery is to be found in the Scottish Highlands."
    #text = "Brain waves during REM sleep appear similar to brain waves during wakefulness."
    #text = "The entire eastern portion of the Aral sea has become a sand desert, complete with the deteriorating hulls of abandoned fishing vessels."
    #text = "He says that you like to swim."
    #text = "Being able to link computers into networks has enormously boosted their capabilities."
    #text = "Monetary policy should be countercyclical to counterbalance the business cycles of economic downturns and upswings."
    #text = "I cheated my girlfriend."
    #text = 'She looks very beautiful.'
    #text = 'I go to school twice a day.'
    #text = "I do not like to read science fiction."
    #text = "The 1828 campaign was unique because of the party organization that promoted Jackson."
    #text = "In a disputed 1985 ruling , the Commerce Commission said Commonwealth Edison could raise its electricity rates by $ 49 million to pay for the plant."
    #text = "The strength of the depletion zone electric field increases as the reverse-bias voltage increases."
    #text = "The first CCTV system was installed by Siemens AG at Test Stand VII in PeenemA_nde, Germany in 1942, for observing the launch of V-2 rockets."

    text = expand_contractions(text)
    print ("\n")
    print ("Preprocessed text: "+ text)
    textList = []
    textList.append(text)
    annotator = Annotator("/Users/onur/Downloads/Qg-Iztech-master/pntl/practnlptools", "/Users/onur/Downloads/Qg-Iztech-master/pntl", "edu.stanford.nlp.trees.")
    #annotations = annotator.get_batch_annotations(textList, dep_parse=True)[0]
    try:
        posTags = annotator.get_annoations(textList, dep_parse=False)['pos']
        chunks = annotator.get_annoations(textList, dep_parse=False)['chunk']
    except IndexError:
        emptyList = []
        return emptyList
    #print ("chunks "+str(chunks))
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")
    srlResult = predictor.predict_json({"sentence": text})

    srls = []
    try:
        for i in range(0,len(srlResult['verbs'])):
            myDict = {}
            description = srlResult['verbs'][i]['description']
            while (find_between(description,"[","]") != ""):
                parts = find_between(description,"[","]").split(": ")
                myDict[parts[0]] = parts[1]
                description = description.replace("["+find_between(description,"[","]")+"]", "")
                if (find_between(description,"[","]") == ""):
                    srls.append(myDict)
    except IndexError:
        emptyList = []
        return emptyList
    #print("srls: "+str(srls))

    nlp = spacy.load('en')
    from spacy.symbols import nsubj
    doc = nlp(u''+text)
    for word in doc:
        print(word.text, word.pos_, word.dep_, word.head.text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    dativeWord = []
    dativeVerb = []
    dativeSubType = []
    dobjWord = []
    dobjVerb = []
    dobjSubType = []
    acompWord = []
    acompVerb = []
    acompSubType = []
    attrWord = []
    attrVerb = []
    attrSubType = []
    pcompPreposition = []
    pcompWord = []
    pcompSubType = []
    dateWord = []
    dateSubType = []
    numWord = []
    numSubType = []
    personWord = []
    personSubType = []
    whereWord = []
    whereSubType = []
    foundQuestions = []
    idiomJson = json.loads(open('idiom.json').read())
    for word in doc:
        if word.dep_ == 'dobj' or word.dep_ == 'ccomp' or word.dep_ == 'xcomp' or word.dep_ == 'dative' or word.dep_ == 'acomp' or word.dep_ == 'attr' or word.dep_ == "oprd":
            try:
                baseFormVerb = getBaseFormOfVerb(word.head.text)
                if word.text.find(idiomJson[baseFormVerb]) != -1: continue
            except KeyError:
                pass
        #what question / who question
        if word.dep_ == 'dobj' or word.dep_ == 'ccomp' or word.dep_ == 'xcomp':
            dobjVerb.append(word.head.text)
            dobjWord.append(word.text)
            dobjSubType.append('dobj')
        if word.dep_ == 'dative':
            dativeVerb.append(word.head.text)
            dativeWord.append(word.text)
            dativeSubType.append('dative')
        if word.dep_ == 'acomp':
            acompVerb.append(word.head.text)
            acompWord.append(word.text)
            acompSubType.append(word.dep_)
        if word.dep_ == 'attr' or word.dep_ == "oprd":
            attrVerb.append(word.head.text)
            attrWord.append(word.text)
            attrSubType.append('attr')
        #what question
        if word.dep_ == 'pcomp':
            pcompPreposition.append(word.head.text)
            pcompWord.append(word.text)
            pcompSubType.append(word.dep_)
    for ent in doc.ents:
        #when question
        if (ent.label_ == 'DATE' and ent.text.find('year old') == -1 and ent.text.find('years old') == -1 ):
            dateWord.append(ent.text)
            dateSubType.append(ent.label_)
        #how many question
        if ent.label_ == 'CARDINAL':
            numWord.append(ent.text)
            numSubType.append(ent.label_)
        #who question
        if ent.label_ == 'PERSON':
            personWord.append(ent.text)
            personSubType.append(ent.label_)
        #where question
        if ent.label_ == 'FACILITY' or ent.label_ == 'ORG' or ent.label_ == 'GPE' or ent.label_ == 'LOC':
            whereWord.append(ent.text)
            whereSubType.append('LOC')

    #Beginning of deconstruction stage      
    for i in range(0,len(dativeWord)):
        for k in range(0,len(dobjWord)):
            if dobjVerb[k] != dativeVerb[i]: continue
            for j in range(0,len(srls)):
                foundSubject = checkForAppropriateObjOrSub(srls,j,0)
                foundObject = checkForAppropriateObjOrSub(srls,j,1)
                foundIndirectObject = checkForAppropriateObjOrSub(srls,j,2)
                if (foundSubject == '') or (foundObject == '')  or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
                if (foundIndirectObject == '')  or (foundIndirectObject == foundObject)  or (foundIndirectObject == foundSubject): continue
                elif (srls[j]['V'] == dobjVerb[k]) and (foundObject.find(dobjWord[k]) != -1 ) and (foundIndirectObject.find(dativeWord[i]) != -1 ):
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    predicates.append(totalPredicates)
                    objects.append(foundIndirectObject + " " + foundObject)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                        extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
                    if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                        extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
                    extraFields.append(extraFieldsString)
                    types.append(dativeSubType[i])
      
    for i in range(0,len(dobjWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            elif (srls[j]['V'] == dobjVerb[i]) and (foundObject.find(dobjWord[i]) != -1 ) :
                index =1 -1
                totalPredicates = srls[j]['V']
                for k in range(0,len(chunks)):
                    found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                    if found:
                        index = k
                for k in range(0,index):
                    reversedIndex = index -1 -k
                    if reversedIndex == -1: break
                    resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                    try:
                        if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                            result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                            if foundSubject.find(result.group(1)) != -1: break
                            totalPredicates = result.group(1) + ' ' + totalPredicates 
                        else: break
                    except AttributeError:
                        break
                nextIndex = index + 1
                if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                    if (nextIndex < len(chunks)):
                        resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                        try:
                            if resultType.group(1) == 'S-PRT':
                                result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates
                        except AttributeError:
                            pass
                if totalPredicates[:3] == 'to ':
                    totalPredicates= totalPredicates[3:]
                predicates.append(totalPredicates)
                objects.append(foundObject)
                subjects.append(foundSubject)
                extraFieldsString = ''
                if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                    extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
                if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                    extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
                extraFields.append(extraFieldsString)
                types.append(dobjSubType[i])

    for i in range(0,len(acompWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundObject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            elif (srls[j]['V'] == acompVerb[i]) and (foundObject.find(acompWord[i]) != -1 ) :
                predicates.append('indicate')
                objects.append(foundObject)
                subjects.append(foundSubject)
                extraFields.append(keyCheck('ARGM-LOC',srls[j],'') + ' ' + keyCheck('ARGM-TMP',srls[j],''))
                types.append(acompSubType[i])

    for i in range(0,len(attrWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            if (foundSubject == '') or (keyCheck('V', srls[j], "") == ""): continue
            for key, value in srls[j].items():
                if (srls[j]['V'] == attrVerb[i] and (value.find(attrWord[i]) != -1 ) and key != "V" and value != foundSubject):
                    predicates.append('describe')
                    objects.append(foundSubject)
                    subjects.append('you')
                    extraFields.append(keyCheck('ARGM-LOC',srls[j],'') + ' ' + keyCheck('ARGM-TMP',srls[j],''))
                    types.append(attrSubType[i])

    for i in range(0,len(dateWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            for key, value in srls[j].items():
                if (value.find(dateWord[i]) != -1 ) and key != "V" and key!= "ARGM-TMP" and value != foundSubject and value != foundObject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    predicates.append(totalPredicates)
                    objects.append(foundObject)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                        extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
                    extraFieldsString = extraFieldsString.replace(dateWord[i], "")
                    extraFields.append(extraFieldsString)
                    types.append(dateSubType[i])

    for i in range(0,len(whereWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            for key, value in srls[j].items():
                if (value.find(whereWord[i]) != -1 ) and key != "V" and key!= "ARGM-LOC" and value != foundSubject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]

                    realObj = ''
                    if(foundObject != '' and value == foundObject):
                        valueArray = value.split(' ')
                        for l in range(0,len((doc))):
                            if(l + 1 >= len(doc)): break
                            if(value.find(doc[l].text) == -1) or (value.find(doc[l+1].text) == -1): continue

                            if whereWord[i].find(doc[l+1].text) != -1 and doc[l].pos_ == 'ADP':
                                break
                            
                            if(realObj == ''): realObj = doc[l].text
                            else: realObj += ' ' + doc[l].text 
                    else:
                        realObj = foundObject

                    predicates.append(totalPredicates)
                    if realObj[-4:] == ' the':
                        realObj = realObj[:-4]
                    objects.append(realObj)
                    subjects.append(foundSubject)
                    extraFieldsString = ''
                    if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                        extraFieldsString = keyCheck('ARGM-TMP',srls[j],'') 
                    extraFields.append(extraFieldsString)
                    types.append(whereSubType[i])

    for i in range(0,len(pcompWord)):
        for j in range(0,len(srls)):
            if(keyCheck('V', srls[j], "") == ''): continue
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            index =1 -1
            totalPredicates = pcompPreposition[i]
            
            for k in range(0,len(chunks)):
                found = re.compile(pcompPreposition[i], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                if found:
                    index = k
            isMainVerbFound = False
            for k in range(0,index):
                reversedIndex = index -1 -k
                if reversedIndex == -1: break
                if (isMainVerbFound == False):
                    totalPredicates= str(chunks[reversedIndex][0]) + ' '+totalPredicates 
                    if (chunks[reversedIndex][0] == srls[j]['V']): 
                        isMainVerbFound = True
                elif (isMainVerbFound == True):
                    resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                    try:
                        if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                            result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                            if foundSubject.find(result.group(1)) != -1: break
                            totalPredicates = result.group(1) + ' ' + totalPredicates 
                        else: break
                    except AttributeError:
                        break
            if (totalPredicates.find(srls[j]['V']) != -1 ):
                if (checkForAppropriateObjOrSub(srls,j,0) != ''):
                    foundObject = checkForAppropriateObjOrSub(srls,j,1)
                else:
                    continue
                for key, value in srls[j].items():
                    if(key != 'V' and value.find(pcompWord[i]) != -1 and foundSubject != ''):
                        if (foundSubject == value and foundObject != '' and foundObject != foundSubject ):
                            subjects.append(foundObject)
                        else:
                            subjects.append(foundSubject)

                        objects.append('')
                        predicates.append(totalPredicates)
                        extraFields.append('')
                        types.append(pcompSubType[i])
            else: 
                continue

    for i in range(0,len(numWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            for key, value in srls[j].items():
                if (value.find(numWord[i]) != -1 ) and key != "V":
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]
                    """
                    srlWordArray = []
                    srlTypeArray = []
                    for word in doc:
                        if (value.find(word.text) != -1):
                            srlWordArray.append(word.text)
                            srlTypeArray.append(word.pos_)
                    numFoundIndex = -1
                    nouns = ''
                    print("aa"+str(srlWordArray))
                    print("bb"+str(srlTypeArray))
                    for l in range(0,len(srlWordArray)):
                        if (srlWordArray[l].find(numWord[i]) != -1):
                            numFoundIndex = l
                            print(l)
                        elif l > numFoundIndex and srlTypeArray[l] == 'NOUN' and numFoundIndex != -1:
                            print("ff " + srlWordArray[l])
                            if(nouns == ''):
                                nouns = srlWordArray[l]
                            else: 
                                nouns += ' '+srlWordArray[l]
                        elif (numFoundIndex != -1): break
                    print("nn "+nouns)
                    #elif (value == foundSubject and foundObject == ''):
                    #elif (value == foundSubject and foundObject != ''):
                    """
                    midFoundIndex = -1
                    valueArray = value.split(" ")
                    for l in range(0,len(valueArray)):
                        if valueArray[l].find(numWord[i]) != -1:
                            midFoundIndex = l

                    valueArrayFirstPart = valueArray[(midFoundIndex + 1):]
                    valueArrayLastPart = valueArray[:midFoundIndex]

                    valueFirstPart = ""
                    for l in range(0,len(valueArrayFirstPart)):
                        if valueFirstPart == "": valueFirstPart = valueArrayFirstPart[l]
                        else: valueFirstPart = valueFirstPart + " " + valueArrayFirstPart[l]

                    valueLastPart = ""
                    for l in range(0,len(valueArrayLastPart)):
                        if valueLastPart == "": valueLastPart = valueArrayLastPart[l]
                        elif l == (len(valueArrayLastPart) -1) and valueArrayLastPart[l] == "the":  break
                        else: valueLastPart = valueLastPart + " " + valueArrayLastPart[l]
                    
                    #if (nouns == ""): continue
                    predicates.append(totalPredicates)
                    subjects.append(valueFirstPart)
                    #extraFields.append(valueLastPart)
                    #objects.append(foundObject + " " + keyCheck('ARGM-LOC',srls[j],'') + " " + keyCheck('ARGM-TMP',srls[j],''))
                    types.append(numSubType[i])

                    extraFieldsString = ''
                    if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                        extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
                    if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                        extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 

                    if (value == foundObject and foundSubject != ''):
                        objects.append(extraFieldsString)
                        extraFields.append(foundSubject)
                    else:
                        objects.append(valueLastPart + " " + foundObject + " " + extraFieldsString)
                        extraFields.append('')
  

    for i in range(0,len(personWord)):
        for j in range(0,len(srls)):
            foundSubject = checkForAppropriateObjOrSub(srls,j,0)
            foundObject = checkForAppropriateObjOrSub(srls,j,1)
            if (foundSubject == '') or (foundSubject == foundObject) or (keyCheck('V', srls[j], "") == ""): continue
            for key, value in srls[j].items():
                if (value.find(personWord[i]) != -1 ) and key != "V" and value == foundSubject and value != foundObject:
                    index =1 -1
                    totalPredicates = srls[j]['V']
                    for k in range(0,len(chunks)):
                        found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
                        if found:
                            index = k
                    for k in range(0,index):
                        reversedIndex = index -1 -k
                        if reversedIndex == -1: break
                        resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
                        try:
                            if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                                result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                                if foundSubject.find(result.group(1)) != -1: break
                                totalPredicates = result.group(1) + ' ' + totalPredicates 
                            else: break
                        except AttributeError:
                            break
                    nextIndex = index + 1
                    if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
                        if (nextIndex < len(chunks)):
                            resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                            try:
                                if resultType.group(1) == 'S-PRT':
                                    result = re.search("\('(.*)',", str(chunks[nextIndex]))
                                    if foundSubject.find(result.group(1)) != -1: break
                                    totalPredicates = result.group(1) + ' ' + totalPredicates
                            except AttributeError:
                                pass
                    if totalPredicates[:3] == 'to ':
                        totalPredicates= totalPredicates[3:]

                    relativeClauseDet = False
                    otherNounsDet = False
                    if (find_between(value,personWord[i],",") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"who") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"that") == " "): relativeClauseDet = True
                    if (find_between(value,personWord[i],"whose") == " "): relativeClauseDet = True
                    if (relativeClauseDet == False):
                        modifSrl = value.replace("' "+personWord[i]+" '", '')
                        modifSrl = value.replace('" '+personWord[i]+' "', '')
                        modifSrl = value.replace(personWord[i], '')
                        modifSrl = modifSrl.split(' ')
                        for m in range(0,len(modifSrl)):
                            for k in range(0,len(chunks)):
                                resultType = re.search("', '(.*)'\)", str(chunks[k]))
                                resultType1 = re.search("'(.*)',", str(chunks[k]))
                                try:
                                    if (modifSrl[m] == resultType1.group(1) and len(resultType1.group(1)) > 1):
                                        if resultType.group(1) == 'B-NP' or resultType.group(1) == 'E-NP' or resultType.group(1) == 'I-NP' or resultType.group(1) == 'S-NP':
                                            otherNounsDet = True
                                            break
                                except AttributeError:
                                    pass
                    extraFieldsString = ''
                    if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                        extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
                    if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                        extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'')
                    if (otherNounsDet == False):
                        extraFields.append(extraFieldsString)
                        types.append(personSubType[i])
                        predicates.append(totalPredicates)
                        objects.append(foundObject)
                        subjects.append('')
                        

    for j in range(0,len(srls)):
        foundSubject = checkForAppropriateObjOrSub(srls,j,0)
        foundObject = checkForAppropriateObjOrSub(srls,j,1)
        if (foundSubject == '') or (keyCheck('V', srls[j], "") == "") or (foundSubject == foundObject): continue
        index =1 -1
        totalPredicates = srls[j]['V']
        for k in range(0,len(chunks)):
            found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
            if found:
                index = k
        for k in range(0,index):
            reversedIndex = index -1 -k
            if reversedIndex == -1: break
            resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
            try:
                if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                    result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                    if foundSubject.find(result.group(1)) != -1: break
                    totalPredicates = result.group(1) + ' ' + totalPredicates 
                else: break
            except AttributeError:
                break
        nextIndex = index + 1
        if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
            if (nextIndex < len(chunks)):
                resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                try:
                    if resultType.group(1) == 'S-PRT':
                        result = re.search("\('(.*)',", str(chunks[nextIndex]))
                        if foundSubject.find(result.group(1)) != -1: break
                        totalPredicates = result.group(1) + ' ' + totalPredicates
                except AttributeError:
                    pass
        if totalPredicates[:3] == 'to ':
            totalPredicates= totalPredicates[3:]

        if (keyCheck('ARGM-CAU', srls[j], '') != ""):
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
            if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
            extraFields.append(extraFieldsString)
            types.append('why')
        if (keyCheck('ARGM-PNC', srls[j], '') != ""):
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
            if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
            extraFields.append(extraFieldsString)
            types.append('purpose')
        if (keyCheck('ARGM-MNR', srls[j], '') != ""):
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
            if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
            extraFields.append(extraFieldsString)
            types.append('how')
        if (keyCheck('ARGM-TMP', srls[j], '') != ""):
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
                extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
            extraFields.append(extraFieldsString)
            types.append('DATE')
        if (keyCheck('ARGM-LOC', srls[j], '') != ""):
            predicates.append(totalPredicates)
            objects.append(foundObject)
            subjects.append(foundSubject)
            extraFieldsString = ''
            if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
                extraFieldsString = keyCheck('ARGM-TMP',srls[j],'') 
            extraFields.append(extraFieldsString)
            types.append('LOC')

    for j in range(0,len(srls)):
        foundSubject = checkForAppropriateObjOrSub(srls,j,0)
        foundObject = checkForAppropriateObjOrSub(srls,j,1)
        if (foundSubject == '') or (keyCheck('V', srls[j], "") == "") or (foundSubject == foundObject) or (foundObject == ""): continue
        index =1 -1
        totalPredicates = srls[j]['V']
        for k in range(0,len(chunks)):
            found = re.compile(srls[j]['V'], re.DOTALL |  re.IGNORECASE).findall(str(chunks[k]))
            if found:
                index = k
        for k in range(0,index):
            reversedIndex = index -1 -k
            if reversedIndex == -1: break
            resultType = re.search("', '(.*)'\)", str(chunks[reversedIndex]))
            try:
                if resultType.group(1) == 'B-VP' or resultType.group(1) == 'E-VP' or resultType.group(1) == 'I-VP' or resultType.group(1) == 'S-VP':
                    result = re.search("\('(.*)',", str(chunks[reversedIndex]))
                    if foundSubject.find(result.group(1)) != -1: break
                    totalPredicates = result.group(1) + ' ' + totalPredicates 
                else: break
            except AttributeError:
                break
        nextIndex = index + 1
        if (srls[j]['V'] != 'am' and srls[j]['V'] != 'is' and srls[j]['V'] != 'are' and srls[j]['V'] != 'was' and srls[j]['V'] != 'were'  and srls[j]['V'] != 'be'):
            if (nextIndex < len(chunks)):
                resultType = re.search("', '(.*)'\)", str(chunks[nextIndex]))
                try:
                    if resultType.group(1) == 'S-PRT':
                        result = re.search("\('(.*)',", str(chunks[nextIndex]))
                        if foundSubject.find(result.group(1)) != -1: break
                        totalPredicates = result.group(1) + ' ' + totalPredicates
                except AttributeError:
                    pass
        if totalPredicates[:3] == 'to ':
            totalPredicates= totalPredicates[3:]

        predicates.append(totalPredicates)
        objects.append(foundObject)
        subjects.append(foundSubject)
        extraFieldsString = ''
        if keyCheck('ARGM-LOC',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-LOC',srls[j],'')) == -1):
            extraFieldsString = keyCheck('ARGM-LOC',srls[j],'')
        if keyCheck('ARGM-TMP',srls[j],'') != '' and (totalPredicates.find(keyCheck('ARGM-TMP',srls[j],'')) == -1):
            extraFieldsString = extraFieldsString + ' ' + keyCheck('ARGM-TMP',srls[j],'') 
        extraFields.append(extraFieldsString)
        types.append('direct')

    print ('---- Found Deconstruction results : ----')
    for i in range(0,len(subjects)):
        print(subjects[i])
        print(predicates[i])
        print(objects[i])
        print(extraFields[i])
        print(types[i])
        print ('----------------------------------------')

    #Beginning of construction stage    
    for i in range(0,len(subjects)):
        negativePart = ''
        negativeIndex = -1
        predArr = predicates[i].split(' ')
        numOfVerbs = 0
        firstFoundVerbIndex = -1
        isToDetected = False
        isAndDetected = False
        for j in range(0,len(predArr)):
            if predArr[j] == 'and': isAndDetected = True
            for k in range (0,len(posTags)):
                if posTags[k][0]== predArr[j] and posTags[k][0] == 'to':
                    isToDetected = True
                if posTags[k][0]== predArr[j] and (posTags[k][1] == 'VB' or posTags[k][1] == 'VBD' or posTags[k][1] == 'VBG' or posTags[k][1] == 'VBN' or posTags[k][1] == 'VBP' or posTags[k][1] == 'VBZ' or posTags[k][1] == 'MD'):
                    if numOfVerbs == 0:
                        firstFoundVerbIndex = j
                    numOfVerbs = numOfVerbs + 1
                    break
                if posTags[k][0]== predArr[j] and posTags[k][1] == 'RB' and posTags[k][0].lower() == 'not':
                    if numOfVerbs == 0:
                        firstFoundVerbIndex = j
                    numOfVerbs = numOfVerbs + 1
                    negativeIndex = j
                    break
                #elif firstFoundVerbIndex != -1: break
        if isAndDetected == False:
            if negativeIndex > -1:
                negativePart = predArr.pop(negativeIndex)
            if numOfVerbs == 1 or isToDetected == True:
                if predArr[0] != 'am' and predArr[0] != 'is' and predArr[0] != 'are' and predArr[0] != 'was' and predArr[0] != 'were':
                    for k in range (0,len(posTags)):
                        predArrNew = []
                        if posTags[k][0] == predArr[firstFoundVerbIndex]:
                            predArrNew = []
                            if posTags[k][1] == 'MD':
                                break
                            if posTags[k][1] == 'VBG': 
                                types[i] = '' 
                                break
                            if posTags[k][1] == 'VBZ':
                                predArrNew.append('does')
                                if posTags[k][0] == 'has':
                                    predArrNew.append(posTags[k][0])
                                else:
                                    predArrNew.append(getBaseFormOfVerb(posTags[k][0]))
                            elif posTags[k][1] == 'VBP':
                                predArrNew = []
                                predArrNew.append('do')
                                predArrNew.append(posTags[k][0])
                            elif posTags[k][1] == 'VBD' or posTags[k][1] == 'VBN':
                                predArrNew = []
                                predArrNew.append('did')
                                predArrNew.append(getBaseFormOfVerb(posTags[k][0]))
                            else:
                                predArrNew = []
                                subjectParts = subjects[i].split(' ')
                                isFound = False
                                for l in range (0,len(posTags)):
                                    if isFound == True: break
                                    for m in range (0,len(subjectParts)):
                                        if posTags[l][0] == subjectParts[m]:
                                            if posTags[l][1] == 'NN':
                                                predArrNew.append('does')
                                                predArrNew.append(posTags[k][0])
                                                isFound = True
                                                break
                                            elif posTags[l][1] == 'NNS':
                                                predArrNew.append('do')
                                                predArrNew.append(posTags[k][0])
                                                isFound = True
                                                break
                                        if l == (len(posTags)-1) and m == (len(subjectParts)-1):
                                            predArrNew.append('do')
                                            predArrNew.append(posTags[k][0])
                                            isFound = True
                                            break
                            predArr.pop(firstFoundVerbIndex)
                            predArrTemp = predArr
                            predArr = predArrNew + predArrTemp
                            break
            if numOfVerbs == 0 and len(predArr) == 1 and types[i] != 'attr':
                mainVerb = predArr[0]
                qVerb = ''
                if getBaseFormOfVerb(predArr[0]) == predArr[0]:
                    qVerb = 'do'
                else:
                    qVerb = 'does'

                predArr = []
                predArr.append(qVerb)
                predArr.append(mainVerb)

        if isAndDetected == True: predArr.insert(0, '')
        #print ('No of verb count: '+str(numOfVerbs))
        isQuestionMarkExist = False
        verbRemainingPart = ''
        question = ''
        for k in range (1,len(predArr)):
            verbRemainingPart = verbRemainingPart + ' ' + predArr[k]

        if types[i] == 'dative':
            question = 'What ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        if types[i] == 'dobj' or types[i] == 'pcomp':
            whQuestion = 'What '
            for ent in doc.ents:
                if(ent.text == objects[i] and ent.label_ == 'PERSON'):
                    whQuestion = 'Who '
            question = whQuestion + predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart  + ' ' + extraFields[i]
        elif types[i] == 'DATE':
            question = 'When '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'LOC':
            question = 'Where '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'CARDINAL':
            question = 'How many ' + subjects[i] + ' ' + predArr[0] + ' ' + negativePart + ' ' + extraFields[i]  + ' ' + verbRemainingPart + ' ' + objects[i]
        elif types[i] == 'attr':
            question = 'How would  '+ subjects[i] + ' ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i]
        elif types[i] == 'PERSON':
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = 'Who  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        elif types[i] == 'WHAT':
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = 'What  ' + predArr[0] + ' ' + negativePart + ' ' + verbRemainingPart + ' ' + objects[i] + ' ' + extraFields[i]
        elif types[i] == 'acomp':
            isQuestionMarkExist = True  
            question = 'Indicate characteristics of ' + getObjectPronun(subjects[i])
        elif types[i] == 'direct':
            predArr[0]= predArr[0][:1].upper() + predArr[0][1:]
            if objects[i].endswith('.'):
                objects[i] = objects[i][:-1]
            question = predArr[0] + ' '  + negativePart + ' ' + subjects[i] + ' ' + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'why':
            question = 'Why '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'purpose':
            question = 'For what purpose '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]
        elif types[i] == 'how':
            question = 'How '+predArr[0] + ' ' + negativePart + ' ' + subjects[i] + verbRemainingPart + ' ' + objects[i]  + ' ' + extraFields[i]


        
        isUpperWord = False
        postProcessTextArr = text.split(' ')
        lowerCasedWord = postProcessTextArr[0][0].lower() + postProcessTextArr[0][1:]

        for ent in doc.ents:
            if ent.text.find(postProcessTextArr[0]) != -1 and (ent.label_ == 'PERSON' or ent.label_ == 'FACILITY' or ent.label_ == 'GPE' or ent.label_ == 'ORG'):
                lowerCasedWord = postProcessTextArr[0]

        if lowerCasedWord == 'i': lowerCasedWord = 'I'
        #Postprocess stage for lower casing common nouns, omitting extra spaces and dots
        formattedQuestion = question.replace(postProcessTextArr[0],lowerCasedWord)
        formattedQuestion = ' '.join(formattedQuestion.split())
        formattedQuestion = formattedQuestion.replace(' ,', ',')
        formattedQuestion = formattedQuestion.replace(" 's " , "'s ")
        formattedQuestion = formattedQuestion.replace("s ' " , "s' ")
        quotatedString = re.findall('"([^"]*)"', formattedQuestion)
        quotatedOrgString = re.findall('"([^"]*)"', formattedQuestion)
        for l in range(0,len(quotatedString)):
            if quotatedString[l][0] == " ": quotatedString[l] = quotatedString[l][1:]
            if quotatedString[l][-1] == " ": quotatedString[l] = quotatedString[l][:-1]
            formattedQuestion = formattedQuestion.replace(quotatedOrgString[l], quotatedString[l])

        while (formattedQuestion.endswith(' ')):
            formattedQuestion = formattedQuestion[:-1]
        if formattedQuestion.endswith('.') or formattedQuestion.endswith(','):
            formattedQuestion = formattedQuestion[:-1]
        if formattedQuestion != '':
            if isQuestionMarkExist == False:
                formattedQuestion = formattedQuestion + '?'
            else: 
                formattedQuestion = formattedQuestion + '.'

            #print (formattedQuestion)
            foundQuestions.append(formattedQuestion)

    foundQuestions.sort(key = lambda s: len(s))
    indexer = len(foundQuestions) - 1

    while (indexer > -1):
        if (indexer -1 < 0): break
        if foundQuestions[indexer] == foundQuestions[indexer-1]:
            foundQuestions.remove(foundQuestions[indexer])
        indexer = indexer - 1

    for i in range(0,len(foundQuestions)):
        print (foundQuestions[i])
    return foundQuestions
    
if __name__ == "__main__":
    text = "In 1980, the son of Vincent J. McMahon, Vincent Kennedy McMahon, founded Titan Sports, Inc. and in 1982 purchased Capitol Wrestling Corporation from his father."
    foundQuestions = generate(text)
    #print(foundQuestions)
    """
    # question similarity and bleu score calculation for QGSTEC 2010 data
    from pycorenlp import StanfordCoreNLP
    file = open("boxPlot.txt","w")
    fileSimilarity = open("textSimilarity.txt","w")
    fileSimilarity2 = open("textSimilarity2.txt","w")
    fMain = open('../caption-eval/data2/allReferences.txt','w')
    fMain.close()
    import xml.etree.ElementTree as ET
    tree = ET.parse('TestData_QuestionsFromSentencesHighScores.xml')
    root = tree.getroot()
    for instance in root.iter('instance'):
        foundQuestions = generate(instance.find('text').text)
        allGTQuestions = ''
        allGeneratedQuestions = ''
        for submission in instance.iter('submission'):
            for question in submission.iter('question'):
                if (os.path.isfile('../caption-eval/data2/references' + str(instance.get('id')) + '.txt') == True):
                    with open('../caption-eval/data2/references' + str(instance.get('id')) + '.txt', "a") as rFile:
                        rFile.write('set' + str(instance.get('id')) + '\t' + question.text.strip() + "\n")
                else:
                    f1 = open('../caption-eval/data2/references' + str(instance.get('id')) + '.txt','w')
                    f1.write('set' + str(instance.get('id')) + '\t' + question.text.strip() + "\n")
                    f1.close()
                with open('../caption-eval/data2/allReferences.txt', "a") as mFile:
                    mFile.write('set' + str(instance.get('id')) + '\t' + question.text.strip() + "\n")
                if allGTQuestions == '':
                    allGTQuestions = question.text.strip()
                else:
                    allGTQuestions = allGTQuestions +  ' | ' + question.text.strip()

                mostMatchedScore = -1
                mostMatchedSentence = ''
                syntacticScore = "0"
                if len(foundQuestions) == 0:
                    print('Final Matched score: 0')
                    print ('----------------------------------------\n')
                    file.write('0' + '\n')
                    notFoundString = "not found"+'\n' + question.text+'\n' + 'semantic similarity:0 ' + '\n' + '------------------------------- \n\n'
                    fileSimilarity.write(notFoundString + '\n')

                for i in range (0,len(foundQuestions)):
                    nlp = spacy.load('en')
                    doc1 = nlp(u''+foundQuestions[i])
                    doc2 = nlp(u''+question.text)
                    for doc in [doc1]:
                        for other_doc in [doc2]:
                            if doc.similarity(other_doc) > mostMatchedScore:
                                mostMatchedScore = doc.similarity(other_doc)
                                mostMatchedSentence = foundQuestions[i]+'\n' + question.text+'\n' + 'semantic similarity: '+ str(doc.similarity(other_doc)) + '\n' + '------------------------------- \n\n'
                                print('Semantic similarity: ' + str(doc.similarity(other_doc)))
                                print('Syntactic similarity: ' + str(syntacticScore))
                                print("Sentence: "+instance.find('text').text)
                                print("Generated question: "+foundQuestions[i])
                                print("Predefined question: "+question.text)
                                print ('----------------------------------------\n')
                    if i == (len(foundQuestions) - 1):
                        print('Final Matched score: ' + str(mostMatchedScore))
                        print ('----------------------------------------\n')
                        file.write(str(mostMatchedScore) + '\n')
                        fileSimilarity.write(mostMatchedSentence + '\n')

        for i in range (0,len(foundQuestions)):
            f2 = open('../caption-eval/data2/predictes' + str(instance.get('id')) + '-'  + str(i) + '.txt','w')
            f2.write('set' + str(instance.get('id')) + '\t' + foundQuestions[i] + "\n")
            f2.close()
            if allGeneratedQuestions == '':
                allGeneratedQuestions = foundQuestions[i]
            else:
                allGeneratedQuestions = allGeneratedQuestions +  ' | ' + foundQuestions[i]

        fileSimilarity2.write(str(instance.get('id')) + ' | ' + instance.find('text').text + '\n')
        fileSimilarity2.write(allGTQuestions + '\n')
        fileSimilarity2.write(allGeneratedQuestions + '\n')

    file.close()
    fileSimilarity.close()
    fileSimilarity2.close()


    # question similarity score calculation for QGSTEC 2010 data
    from pycorenlp import StanfordCoreNLP
    nlpStanford = StanfordCoreNLP('http://localhost:9000')
    file = open("boxPlot.txt","w")
    fileSimilarity = open("textSimilarity.txt","w")
    import xml.etree.ElementTree as ET
    tree = ET.parse('TestData_QuestionsFromSentencesHighScores.xml')
    root = tree.getroot()
    for instance in root.iter('instance'):
        foundQuestions = generate(instance.find('text').text)
        for submission in instance.iter('submission'):
            for question in submission.iter('question'):
                mostMatchedScore = -1
                mostMatchedSentence = ''
                syntacticScore = "0"
                if len(foundQuestions) == 0:
                    print('Final Matched score: 0')
                    print ('----------------------------------------\n')
                    file.write('0' + '\n')
                    notFoundString = "not found"+'\n' + question.text+'\n' + 'semantic similarity:0 ' + '\n' + 'syntactic similarity: 0'+ '\n' + '------------------------------- \n\n'
                    fileSimilarity.write(notFoundString + '\n')
                for i in range (0,len(foundQuestions)):
                    nlp = spacy.load('en')
                    doc1 = nlp(u''+foundQuestions[i])
                    doc2 = nlp(u''+question.text)
                    for doc in [doc1]:
                        for other_doc in [doc2]:
                            if doc.similarity(other_doc) > mostMatchedScore:
                                #annotator=Annotator()
                                #syntacticScore = tree_kernel.calculateSynacticSimilarity(annotator.getAnnotations(question.text,dep_parse=False)['syntax_tree'],annotator.getAnnotations(foundQuestions[i],dep_parse=False)['syntax_tree'])
                                text = question.text
                                output = nlpStanford.annotate(text, properties={
                                  'annotators': 'parse',
                                  'outputFormat': 'json'
                                })

                                text1 = str(foundQuestions[i])
                                output1 = nlpStanford.annotate(text1, properties={
                                  'annotators': 'parse',
                                  'outputFormat': 'json'
                                })

                                outputArr = output['sentences'][0]['parse'].split()
                                clearOutput = ''
                                for j in range(0, len(outputArr)):
                                    if outputArr[j][-1:] == ')':
                                        clearOutput = clearOutput + ' ' + str(outputArr[j])
                                    else:
                                        clearOutput = clearOutput + str(outputArr[j])

                                outputArr1 = output1['sentences'][0]['parse'].split()
                                clearOutput1 = ''
                                for j in range(0, len(outputArr1)):
                                    if outputArr1[j][-1:] == ')':
                                        clearOutput1 = clearOutput1 + ' ' + str(outputArr1[j])
                                    else:
                                        clearOutput1 = clearOutput1 + str(outputArr1[j])

                                syntacticScore = tree_kernel.calculateSynacticSimilarity(clearOutput,clearOutput1)
                                mostMatchedScore = doc.similarity(other_doc)
                                mostMatchedSentence = foundQuestions[i]+'\n' + question.text+'\n' + 'semantic similarity: '+ str(doc.similarity(other_doc)) + '\n' + 'syntactic similarity: '+ str(syntacticScore) + '\n' + '------------------------------- \n\n'
                                print('Semantic similarity: ' + str(doc.similarity(other_doc)))
                                print('Syntactic similarity: ' + str(syntacticScore))
                                print("Sentence: "+instance.find('text').text)
                                print("Generated question: "+foundQuestions[i])
                                print("Predefined question: "+question.text)
                                print ('----------------------------------------\n')
                    if i == (len(foundQuestions) - 1):
                        print('Final Matched score: ' + str(mostMatchedScore))
                        print ('----------------------------------------\n')
                        file.write(str(mostMatchedScore) + '\n')
                        fileSimilarity.write(mostMatchedSentence + '\n')

    # question similarity score calculation of H&S system for QGSTEC 2010 data
    from pycorenlp import StanfordCoreNLP
    nlpStanford = StanfordCoreNLP('http://localhost:9000')
    file = open("boxPlotH&S.txt","w")
    fileSimilarity = open("textSimilarityH&S.txt","w")
    import xml.etree.ElementTree as ET
    tree = ET.parse('TestData_QuestionsFromSentencesHighScores.xml')
    root = tree.getroot()
    for instance in root.iter('instance'):
        for submission in instance.iter('submission'):
            for question in submission.iter('question'):
                foundQuestions = ""
                with open('H&S-GeneratedQuestions.txt', 'r') as myfile:
                    data = myfile.read().replace('\n', '')
                    generatedQuestionsOld = str(find_between(data, instance.get('id') + " ||", " | || "))
                    foundQuestions = generatedQuestionsOld.split(" | ")
                mostMatchedScore = -1
                mostMatchedSentence = ''
                if len(foundQuestions) == 0 or len(foundQuestions) == 1 or "[" in foundQuestions[0]:
                    print('Final Matched score: 0')
                    print ('----------------------------------------\n')
                    notFoundString = "not found"+'\n' + question.text+'\n' + 'semantic similarity:0 ' + '\n' + 'syntactic similarity: 0'+ '\n' + '------------------------------- \n\n'
                    fileSimilarity.write(notFoundString + '\n')
                    file.write('0' + '\n')
                else:
                    for i in range (0,len(foundQuestions)):
                        nlp = spacy.load('en')
                        doc1 = nlp(u''+foundQuestions[i])
                        doc2 = nlp(u''+question.text)
                        for doc in [doc1]:
                            for other_doc in [doc2]:
                                if doc.similarity(other_doc) > mostMatchedScore:
                                    #annotator=Annotator()
                                    #syntacticScore = tree_kernel.calculateSynacticSimilarity(annotator.getAnnotations(question.text,dep_parse=False)['syntax_tree'],annotator.getAnnotations(foundQuestions[i],dep_parse=False)['syntax_tree'])
                                    text = question.text
                                    output = nlpStanford.annotate(text, properties={
                                      'annotators': 'parse',
                                      'outputFormat': 'json'
                                    })

                                    text1 = str(foundQuestions[i])
                                    output1 = nlpStanford.annotate(text1, properties={
                                      'annotators': 'parse',
                                      'outputFormat': 'json'
                                    })

                                    outputArr = output['sentences'][0]['parse'].split()
                                    clearOutput = ''
                                    for j in range(0, len(outputArr)):
                                        if outputArr[j][-1:] == ')':
                                            clearOutput = clearOutput + ' ' + str(outputArr[j])
                                        else:
                                            clearOutput = clearOutput + str(outputArr[j])

                                    outputArr1 = output1['sentences'][0]['parse'].split()
                                    clearOutput1 = ''
                                    for j in range(0, len(outputArr1)):
                                        if outputArr1[j][-1:] == ')':
                                            clearOutput1 = clearOutput1 + ' ' + str(outputArr1[j])
                                        else:
                                            clearOutput1 = clearOutput1 + str(outputArr1[j])

                                    syntacticScore = tree_kernel.calculateSynacticSimilarity(clearOutput,clearOutput1)
                                    mostMatchedScore = doc.similarity(other_doc)
                                    mostMatchedSentence = foundQuestions[i]+'\n' + question.text+'\n' + 'semantic similarity: '+ str(doc.similarity(other_doc)) + '\n' + 'syntactic similarity: '+ str(syntacticScore) + '\n' + '------------------------------- \n\n'
                                    print('Semantic similarity: ' + str(doc.similarity(other_doc)))
                                    print('Syntactic similarity: ' + str(syntacticScore))
                                    print("Sentence: "+instance.find('text').text)
                                    print("Generated question: "+foundQuestions[i])
                                    print("Predefined question: "+question.text)
                                    print ('----------------------------------------\n')
                        if i == (len(foundQuestions) - 1):
                            print('Matched score: ' + str(mostMatchedScore))
                            print ('----------------------------------------\n')
                            file.write(str(mostMatchedScore) + '\n')
                            fileSimilarity.write(mostMatchedSentence + '\n')

    # question bleu, meteor, rouge score calculation for 2017 Neural Question Generation Data
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data = json.load(open('dev.json'))
    fMain = open('../caption-eval/data/allReferences.txt','w')
    fMain.close()
    for i in range(0,len(data)):
        k = 0
        for parag in data[i]['paragraphs']:
            sentenceList = tokenizer.tokenize(parag['context'])
            for qas in parag['qas']:
                totalIndexes = []
                for ans in qas['answers']:
                    searchString = ''
                    for j in range(0,len(sentenceList)):
                        searchString = searchString + sentenceList[j] + ' '
                        if ans['answer_start'] < len(searchString):
                            found = False
                            for p in range(0,len(totalIndexes)):
                                if totalIndexes[p] == j:
                                    found = True
                            if found == False:
                                totalIndexes.append(j)
                            break
                pathString = ''
                for p in range(0,len(totalIndexes)):
                    if pathString == '':
                        pathString = str(totalIndexes[p])
                    else:
                        pathString = pathString + '=' + str(totalIndexes[p])


                if (os.path.isfile('../caption-eval/data/references' + str(i) + '-' + str(k) + '-' + pathString + '.txt') == True):
                    with open('../caption-eval/data/references' + str(i) + '-' + str(k) + '-' + pathString + '.txt', "a") as existFile:
                        existFile.write('set' + str(i) + '-' + str(k) + '-' + pathString + '\t' + qas['question'] + "\n")
                    with open('../caption-eval/data/allReferences.txt', "a") as mFile:
                        mFile.write('set' + str(i) + '-' + str(k) + '-' + pathString + '\t' + qas['question'] + "\n")

                else:
                    questionTotalList = ''
                    for p in range(0,len(totalIndexes)):
                        questionTotalList = questionTotalList + sentenceList[totalIndexes[p]] + ' '

                    questionList = generate(questionTotalList)
                    for l in range (0, len(questionList)):
                        f1 = open('../caption-eval/data/predictes' + str(i) + '-' + str(k) + '-' + pathString + '-' + str(l) + '.txt','w')
                        f1.write('set' + str(i) + '-' + str(k) + '-' + pathString + '\t' + questionList[l] + "\n")
                        f1.close()
                    if len(questionList) > 0:
                        f2 = open('../caption-eval/data/references' + str(i) + '-' + str(k) + '-' + pathString + '.txt','w')
                        f2.write('set' + str(i) + '-' + str(k) + '-' + pathString + '\t' + qas['question'] + "\n")
                        f2.close()
                        with open('../caption-eval/data/allReferences.txt', "a") as mFile:
                            mFile.write('set' + str(i) + '-' + str(k) + '-' + pathString + '\t' + qas['question'] + "\n")

            k = k + 1
    """
    # n-gram individual BLEU
    #from nltk.translate.bleu_score import sentence_bleu
    #reference = [['this', 'is', 'a', 'test'],['this', 'is', 'test']]
    #candidate = ['this', 'is', 'a', 'exam']
    #print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
    #print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
    #print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
    #print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))


    #generate("I hang out at the pool hall.")
    #generate("The Bill of Rights gave the new federal government greater legitimacy.")
    #generate("I hang out with my friends.")
    #generate("i sleep with my eyes open.")
    #generate("Early in the twentieth century, Thorstein Veblen, an American institutional economist, analysed cultural influences on consumption.")
    #generate('The "Rising Sun" owned by Larry Ellison is up for sale.')
    #generate("Kate was put off by the word 'paradox' and Erin did not know what 'marginal tax' meant.")
    #generate("Atop the Main Building's gold dome is a golden statue of the Virgin Mary.")
    #generate("In 1980, the son of Vincent J. McMahon, Vincent Kennedy McMahon, founded Titan Sports, Inc. and in 1982 purchased Capitol Wrestling Corporation from his father.")
    #generate("The first plan to harness the Shannon's power between Lough Derg and Limerick was published in 1844 by Sir Robert Kane.")
    #generate("Morgan was shocked by the reminder of his part in the stock market crash and by Tesla's breach of contract by asking for more funds.")
    """
    output = open("generationOutput.txt","w")
    with open("dataSetNew.txt", "r") as ins:
        for line in ins:
            output.write("S: " + line + '\n')
            foundQuestions = generate(line)
            for j in range(0, len(foundQuestions)):
                output.write("Q: " + foundQuestions[j] + '\n')
            output.write('----------------------------------------------------------------------------\n')
    """
    """
    # question generation data for 2017 Neural Question Generation Data
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data = json.load(open('dev2.json'))
    fileSimilarity3 = open("textSimilarity3.txt","w")
    k = 0
    for i in range(0,len(data)):
        for parag in data[i]['paragraphs']:
            sentenceList = tokenizer.tokenize(parag['context'])
            for qas in parag['qas']:
                for ans in qas['answers']:
                    searchString = ''
                    for j in range(0,len(sentenceList)):
                        searchString = searchString + sentenceList[j] + ' '
                        if ans['answer_start'] < len(searchString):
                            questionList = generate(sentenceList[j])
                            allGeneratedQuestions = ''
                            for l in range (0, len(questionList)):
                                if allGeneratedQuestions == '':
                                    allGeneratedQuestions = questionList[l]
                                else:
                                    allGeneratedQuestions = allGeneratedQuestions +  ' | ' + questionList[l]
                            if len(questionList) > 0:
                                fileSimilarity3.write(str(k) + ' | ' + sentenceList[j] + '\n')
                                fileSimilarity3.write(qas['question'] + '\n')
                                fileSimilarity3.write(allGeneratedQuestions + '\n')
                                k = k + 1
                            break
                    break

    # create datasets for user study
    from random import randint
    with open('chooseQuestion.txt') as f:
        lines = f.read().splitlines()
        for j in range(0,3):
            for i in range(0,50):
                rand = randint(0, 2170)
                with open('chooseQuestion' + str(j) + '.txt', "a") as mFile:
                    mFile.write(str(i) + ' | ' + lines[rand] + '\n')
                    foundQuestions = generate(lines[rand])
                    for k in range(0, len(foundQuestions)):
                        if k == len(foundQuestions) - 1:
                            mFile.write(foundQuestions[k] + '\n')
                        else:
                            mFile.write(foundQuestions[k] + ' | ')
    """
    # evaluate .csv file for user study
    '''import csv
    from collections import defaultdict

    columns = defaultdict(list) # each value in each column is appended to a list
    totalDifficulty = 0
    totalAmbiguity = 0
    totalCorrectness = 0
    totalRelevance = 0
    indexDifficulty = 0
    indexAmbiguity = 0
    indexCorrectness = 0
    indexRelevance = 0
    with open('QG-User-Study-2a.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                if k.find('[Difficulty]') != -1:
                    columns['Difficulty'].append(v)
                    totalDifficulty = totalDifficulty + float(v)
                    indexDifficulty = indexDifficulty + 1
                elif k.find('[Ambiguity]') != -1:
                    columns['Ambiguity'].append(v)
                    totalAmbiguity = totalAmbiguity + float(v)
                    indexAmbiguity = indexAmbiguity + 1
                elif k.find('[Correctness]') != -1:
                    columns['Correctness'].append(v)
                    totalCorrectness = totalCorrectness + float(v)
                    indexCorrectness = indexCorrectness + 1
                elif k.find('[Relevance]') != -1:
                    columns['Relevance'].append(v)
                    totalRelevance = totalRelevance + float(v)
                    indexRelevance = indexRelevance + 1

    print("Total Count: "+str(indexDifficulty))
    print("Average Difficulty: " + str(totalDifficulty / indexDifficulty) )
    print("Average Ambiguity: " + str(totalAmbiguity / indexAmbiguity) )
    print("Average Correctness: " + str(totalCorrectness / indexCorrectness) )
    print("Average Relevance: " + str(totalRelevance / indexRelevance) )
    #print(columns['Difficulty'])
    #print(columns['Ambiguity'])
    #print(columns['Correctness'])
    #print(columns['Relevance'])'''