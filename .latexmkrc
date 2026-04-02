# hsmw-thesis + biblatex + cleveref + totalcount often need >5 passes to clear
# "There were undefined references" and "Rerun to get them right".
$max_repeat = 10;

# Keep root clean: send all generated artifacts to build/
$aux_dir = 'build';
$out_dir = 'build';
$ENV{'TEXMF_OUTPUT_DIRECTORY'} = 'build';

# Keep latexmk non-interactive on errors.
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
