set terminal postscript eps enhanced color dashed "Helvetica" 22

################################
# functinos                    #
################################

# limit data based on aposteriori cycle
extr_cycle(N0,N1,fname)="< awk '$1>=".N0." && $1 <= ".N1."' ".fname

# function to fit error convergence
func_edof(x,a1,a2) = a1+a2*x

# function to plot error convergence
plot_edof(x,a1,a2) = 10**(a1+a2*log10(x))

# a function to get Y2 coordinate of convergence-rate-triangle
# given X1,Y1 (lower left), X2 (right) and a2 from the above function
func_Y2(X1,Y1,X2,a2) = Y1 * 10**(a2 * log10(X1 / X2) );

# a function to get a value of certain column in the last row of a given file
converged_value(Ecl,fname)=system("tail -1 " . fname . " | awk '{ print $" . Ecl . "}'");

# a function to get a value of certain column in the given row of a given file
converged_value_n(Ecl,fname,N)=system("awk '$1==".N." { print $" . Ecl . "}' ".fname);

############################
# line styles              #
############################
# http://colorbrewer2.org/?type=qualitative&scheme=Set1&n=9
# red:
set style line 1  linetype 1 linecolor rgb "#e41a1c"  linewidth 3.000 pointtype 4 pointsize 2.0
# blue:
set style line 2  linetype 1 linecolor rgb "#377eb8"  linewidth 3.000 pointtype 6 pointsize 2.0
# green:
set style line 3  linetype 1 linecolor rgb "#4daf4a"  linewidth 3.000 pointtype 8 pointsize 2.0
# purple:
set style line 4  linetype 1 linecolor rgb "#984ea3"   linewidth 3.000 pointtype 9 pointsize 2.0
# orange:
set style line 5  linetype 5 dt 2 linecolor rgb "#ff7f00"   linewidth 3.000 pointtype 11 pointsize 2.0
# yellow:
set style line 6  linetype 5 dt 3 linecolor rgb "#ffff33"   linewidth 3.000 pointtype 5 pointsize 2.0
# brown
set style line 7  linetype 8 dt 4 linecolor rgb "#a65628"   linewidth 3.000 pointtype 8 pointsize 2.0
# pink
set style line 8  linetype 8 dt 5 linecolor rgb "#f781bf"   linewidth 3.000 pointtype 8 pointsize 2.0
# grey:
set style line 9  linetype 4 linecolor rgb "#999999"    linewidth 4.000 pointtype 1 pointsize 0.0
# black:
set style line 10  linetype 1 linecolor rgb "black"    linewidth 2.000 pointtype 1 pointsize 0.0

##############################
#                            #
#    P L O T T I N G         #
#                            #
##############################

#
Eh = -0.5

HAE = "tests/coulomb_02.mpirun=4.output"

set autoscale xy

set xlabel "DoF"
set ylabel "error"
set logscale xy
set format x "10^{%T}"
set format y "10^{%T}"

# fit
#
HAEa1 = 1;
HAEa2 = -2;
fit func_edof(x,HAEa1,HAEa2) extr_cycle(8,15,HAE) using (log10(column(3))):(log10((column(4)-Eh))) via HAEa1, HAEa2

#set yrange [1e-5:1e+1]
#set xrange [5e+2:1e+5]
set output 'plot.eps'
plot HAE  using (column(3)):(column(4)-Eh) axis x1y1 with lp ls 2 title 'FE solution', \
     plot_edof(x,HAEa1,HAEa2) with l ls 10 dt 4 title sprintf("a line with slope %1.2f",HAEa2)
