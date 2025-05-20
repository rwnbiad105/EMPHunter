# Extended Malicious Packages Hunter
***

This is the scource code and the data sets of EMPHunter.

For PyPI malware detection, the source code is in ./PyPI/EMPHunter-PyPI. Please read README-PyPI.md for more detailed information.

For npm malware detection, the source code is in ./npm/EMPHunter-npm. Please read README-npm.md for more detailed information.

The data can be download on: https://zenodo.org/records/15470290

Please replace the following DataShare directories with this DataShare directories:

npm-Data/DataShare     ==>     /EMPHunter/npm/EMPHunter-npm/DataShare

PyPI-Data/DataShare     ==>     /EMPHunter/PyPI/EMPHunter-PyPI/DataShare


## Abstruct

As the most popular Python software repository, PyPI and npm have become an indispensable part of the Python and JavaScript ecosystem. Regrettably, the open nature of  PyPI and npm exposes end-users to substantial security risks stemming from malicious packages. Consequently, the timely and effective identification of malware within the vast number of newly-uploaded PyPI packages has emerged as a pressing concern. Existing detection methods are dependent on difficult-to-obtain explicit knowledge, such as taint sources, sinks, and malicious code patterns, rendering them susceptible to overlooking emergent malicious packages.

EMPHunter is proposed as a static tool to detect PyPI malicious packages _**without requiring any explicit prior knowledge.**_ EMPHunter utilizes clustering techniques to group the installation scripts of PyPI packages, identifying outliers and ranking them according to their outlierness and the distance between them and known malicious instances, thus highlighting potential evil packages.


## Core Idea

After ana-lyzing some malicious samples, we have found that there are
two exploitable facts.

First, malicious packages are significantly rarer than normal
ones, akin to a needle in a haystack. Although numerous
malicious packages have emerged on PyPI, the vast majority
of PyPI packages remain benign.

Second, the functionality of malicious installation scripts
substantially differs from those of benign ones, with the latter
frequently forming clusters. The setup.py script is dedicated
to configuring and managing the construction, publication,
and installation of packages. Consequently, the activities of
benign setup.py scripts tend to be similar. However, malicious
installation scripts execute dangerous operations that are un-
desirable for a normal setup.py and seldom present in it, such
as accessing sensitive information and installing backdoors.

Based on these observations, we introduce a novel method
called EMPHunter (Malicious Packages Hunter) for detect-
ing malicious PyPI packages. In contrast to existing studies,
EMPHunter directly leverages readily available benign packages
and known malicious samples, avoding the tedious and error-
prone manual analysis needed to extract explicit detection
knowledge.

## How it works

EMPHunter detects malware with following three steps.

In the first step, with assistance from the token embedding model (TEM), we extract canonical token sequences from the installation scripts of the target packages. This ensures that token sequences obtained from scripts with homogeneous semantics are as consistent as possible. The sequences are then encoded using the script embedding model (SEM) to represent the installation scripts of newly uploaded packages and pre-collected benign ones as vectors.

We employed different methods to embed code scripts and shell scripts. Python and JavaScript scripts typically contain rich control information. To effectively represent the structural information in the code, we will traverse the control flow graph (CFG) of the code to extract the canonical sequence of function calls for embedding. Although shell scripts can also include a variety of control structures, shell installation scripts in npm packages are generally quite simple. In fact, many shell installation scripts consist of only a few sequential commands. Therefore, we will extract the command names, options, and parameter texts from the shell scripts in a literal order to form a token sequence for embedding. The canonicalization of tokens will focus on the options within the same command. Additionally, the issue of option name conflicts is prominent in shell commands, while the command parameter values often contain significant noise. This can severely disrupt the embedding of relatively short shell scripts. Consequently, we will apply additional normalization to the shell scripts prior to embedding.

In the second step, the obtained embedding vectors are clustered with the density-based clustering technique to identify outliers, i.e., candidates for malicious installation code scripts.

Finally, candidates are ranked by measuring the distances between them and benign samples, as well as known malicious samples. This enables us to select the most likely suspects for auditing to detect newly-uploaded malicious packages. Candidates distant from benign samples but close to known malicious ones will be considered for manual analysis.