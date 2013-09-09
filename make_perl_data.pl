#!/usr/bin/env perl

use strict;
use warnings;
use Ska::ACACoordConvert qw(toPixels toAngle);

open(my $PA, ">pix_to_angle.txt");
print $PA "row, col, yang, zang\n";
for (my $row = -510; $row <= 510; $row += 42.5){
  for (my $col = -510; $col <= 510; $col += 42.5){
    my ($yang, $zang) = toAngle($row, $col);
    print $PA "$row, $col, $yang, $zang \n";
  }
}
close($PA);

open(my $AP, ">angle_to_pix.txt");
print $AP "yang, zang, row, col\n";
for (my $yang = -2485,; $yang <= 2540; $yang += 209.375){
  for (my $zang = -2545; $zang <= 2501; $zang += 210.25){
    eval {
      my ($row, $col) = toPixels($yang, $zang);
      print $AP "$yang, $zang, $row, $col\n";
    };
    if ($@){
      print "$yang, $zang off CCD\n";
    }
  }
}
close($AP);
