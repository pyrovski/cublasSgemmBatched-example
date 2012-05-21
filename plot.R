#!/usr/bin/env Rscript
pdf('cublas_vs_custom.pdf')
a = read.table('singleTable')
b = read.table('plmLog')
plot(a$V1[1:79], a$V2[1:79], col='red', type='l', lwd=3, xlab='matrix dimension', ylab='time per operation (s)', main='cublasDgemmBatched vs custom kernel time', sub='18000 operations per iteration')
lines(b$V1+1, b$V2, col='black', lwd=3)
legend(x='bottomright', legend=c('cublas', 'custom'), col=c('red', 'black'), lwd=3)
dev.off();
