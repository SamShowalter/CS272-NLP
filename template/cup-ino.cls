\ProvidesClass{cup-ino}[2020/10/20 Cambridge University Press, International Organization Journal submission template v1.2]

\newif\if@suppmat\@suppmattrue
\DeclareOption{nosupp}{\@suppmatfalse}
\ProcessOptions\relax
\ExecuteOptions{}
\LoadClass[twoside,10pt]{article}
\RequirePackage{graphicx}
\RequirePackage{etoolbox}
\RequirePackage{regexpatch}
\RequirePackage{ifxetex,ifluatex}

\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\RequirePackage{microtype}

\ifxetexorluatex
  \RequirePackage[libertine]{newtxmath}
  \RequirePackage[no-math]{fontspec}
  \setmainfont{TeX Gyre Termes}
\else
  \RequirePackage[utf8]{inputenc}
  \RequirePackage[T1]{fontenc}
  \RequirePackage{newtxtext,newtxmath}
\fi

\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{lastpage}
\RequirePackage[medium,small,raggedright,noindentafter]{titlesec}
\RequirePackage[runin]{abstract}
\RequirePackage{booktabs}
\RequirePackage{caption,subcaption}
\RequirePackage{flafter}
\RequirePackage[norule,multiple]{footmisc}
\RequirePackage{enumitem}
\RequirePackage[figuresright]{rotating}
\RequirePackage{lineno}

\geometry{paperwidth=7in,paperheight=9in,
  top=0.5in,bottom=0.5in,inner=0.5in,outer=0.6in,
  headsep=0.1in,footskip=0.3in,footnotesep=0.4in,
  reversemarginpar,marginparwidth=0.575in,marginparsep=0.5em}
  
\newcommand{\runningtitle}[1]{\def\@runningtitle{#1}}
\def\@journalname{CS272 TEMPLATE FOR HW}

\renewcommand{\@maketitle}{%
  \begingroup\raggedright%
  \rule{\textwidth}{1pt}\par
  {\LARGE\bfseries\@title\par}
  \medskip
  {\Large\@author\par}
  \endgroup
}

\renewcommand{\maketitle}{\vspace*{\dimexpr-\headsep-\baselineskip}\@maketitle\thispagestyle{firstpage}}

\titlespacing{name=\section}{0pt}{0pt}{0.5em}
\titlespacing{name=\section,numberless}{0pt}{0pt}{0.1em}
  
\fancyhf{}
\renewcommand{\headrulewidth}{1pt}
\fancyhead[RO]{\@runningtitle\quad\textbf{\thepage}}
\fancyhead[LE]{\textbf{\thepage}\quad\@journalname}
\pagestyle{fancy}
\fancypagestyle{firstpage}{%
  \fancyhf{}
  \renewcommand{\headrule}{}
%   \fancyfoot[L]{\footnotesize\emph{\@journalname}, 20XX, page \thepage\ of \pageref{LastPage}\\%
%   \textcopyright\space The IO Foundation, 20XX}
%   \fancyfoot[R]{\footnotesize~\\doi:10.1017/xxxxxxx xxxx}
}

\setlength{\abstitleskip}{-\parindent}
\abslabeldelim{\quad}
\setlength{\absleftindent}{1em}
\setlength{\absrightindent}{1em}
\preto{\abstract}{%
  {\noindent\rule{\textwidth}{1pt}}\vspace*{-0.8em}%
}
\appto{\endabstract}{%
  \vspace*{-1.2em}%
  {\noindent\raisebox{0.0em}{\rule{\textwidth}{1pt}}}
}

% \preto{\enddocument}{%
%   {\bigskip\noindent\abstracttextfont Date received: MMMM DD, YYYY; Date accepted: MMMM DD, YYYY. \par}\marginpar{\scriptsize\raggedleft Dummy dates; please ignore.}%
% }

\setcounter{secnumdepth}{0}
\setlist{nosep,leftmargin=*,align=left,widest=00}
\newenvironment{authorbio}{\setlength{\parindent}{\z@}}{}

\captionsetup*{labelfont={bf,footnotesize},textfont={it},labelsep={period},justification=centering,singlelinecheck=false}
\captionsetup*[figure]{name=FIGURE}
\captionsetup*[table]{name=TABLE,position=top,justification=raggedright}
\newcommand{\floatnotes}[2][Notes: ]{%
  \par\medskip%
  \begingroup%
  \begin{minipage}{0.9\linewidth}
  \raggedright\footnotesize%
  \emph{#1}#2\par
  \end{minipage}\endgroup%
}

% \RequirePackage[backend=biber,authordate,autocite=footnote,dashed=false,language=british]{biblatex-chicago}
% \ExecuteBibliographyOptions[jurisdiction,legal,legislation]{skipbib=false}
% \DeclareFieldFormat[article]{title}{#1\isdot}
% \DeclareFieldFormat[incollection]{title}{#1\isdot}
% \DeclareFieldFormat[inbook]{title}{#1\isdot}
% \DeclareFieldFormat[report]{title}{#1\isdot}
% \DeclareFieldFormat[inproceedings]{title}{#1\isdot}
% \DeclareFieldFormat[inproceedings]{booktitle}{#1\isdot}
% \DeclareFieldFormat[thesis]{title}{#1\isdot}
% \DeclareFieldFormat[misc]{title}{#1\isdot}
% \DeclareFieldFormat{url}{Available at <\url{#1}>}
% \DefineBibliographyStrings{english}{%
%   in = {Paper presented at the },
% }

% \AtEveryBibitem{%
%   \ifentrytype{book}{\clearlist{location}}{}
% }
% \urlstyle{same}

% \xpatchbibmacro{bib+doi+url}
%   {\printurldate}{}{}{}
% \xapptobibmacro{bib+doi+url}{%
%   \newunit\newblock%
%   \ifboolexpr{%
%     togl {cms@url}%
%     and
%     not test {\iffieldundef{urlyear}}%
%     }%
%   {\printurldate}% Date fix
%   {}%
% }{}{}
% %% Remove space after colon of numbers, before pages
% \xpatchbibmacro*{journal+issue+year+pages}
%   {\setunit{\addcolon\addspace}}
%   {\setunit{\addcolon}}
%   {}{}

% \setlength{\bibhang}{1em}
% \renewcommand{\bibfont}{\small}
% \setlength{\bibitemsep}{\z@}

% \newcommand{\insert@supplmat}{%
% \if@suppmat
%   \section{Supplementary Material}
%   (This is dummy text) Supplementary material for this research note is available at \url{<https://doi.org/10.1017/Sxxxxxxxxx>.}
% \fi\@suppmatfalse
% }
% \xpretocmd{\printbibliography}{\insert@supplmat}{}{}
% \AtBeginEnvironment{thebibliography}{\insert@supplmat}

\appto{\appendix}{%
  \insert@supplmat
  \setcounter{table}{0}
  \setcounter{figure}{0}
  \setcounter{equation}{0}
  \renewcommand{\thefigure}{A\arabic{figure}}
  \renewcommand{\thetable}{A\arabic{table}}
  \renewcommand{\theequation}{A\arabic{equation}}
  \section*{Appendix}
}

%% Tables use smaller font sizes
\DeclareRobustCommand{\patchINOtable}[1]{%
  \BeforeBeginEnvironment{#1}{\footnotesize}
}
%% Heavy table rules
\renewcommand{\belowrulesep}{1em}
\renewcommand{\aboverulesep}{1em}
\setlength{\heavyrulewidth}{2pt}
\AtBeginDocument{%
  \patchINOtable{tabular}
  \patchINOtable{tabular*}
  \patchINOtable{sidewaystable}
  \patchINOtable{sidewaystable*}
  \@ifpackageloaded{tabularx}{\patchINOtable{tabularx}}{}
  \@ifpackageloaded{supertabular}{\patchINOtable{supertabular}}{}
  \@ifpackageloaded{longtable}{\patchINOtable{longtable}\setlength{\LTleft}{\z@}}{}
%   \linenumbers
  \ifdefvoid{\@runningtitle}{\edef\@runningtitle{\@title}}{}
}

\AtEndPreamble{
  \RequirePackage[hyperfootnotes=false]{hyperref}
}
