source("beneficials_accross_backgrounds.R")

# .. This script first plots the DFE os beneficial mutations in the ancestor, and compares it with the corresponding DFE these mutations produce in later generations. Second, it does the reverse analyses (DFE of beneficials in later generations compared with DFE of same mutations in the ancestor)

# .. operations to create a data table in which only cases with values in the three backgrounds are included (make sure to obtain Repi, Tepi and Fepi from 'segments_ben.R' first)
data<-rbind(rbind(Repi,Tepi),Fepi)
data$nas<-rowSums(!is.na(data))
data$back<-c(rep(1,nrow(Repi)), rep(2,nrow(Tepi)), rep(3,nrow(Fepi)))
x<-data[data$nas>=2,]

# .. initialize graphical output
png("overlapped_DFEs.png", width=680*0.011, height=420*0.011, units = 'in', res = 300)
par(family="sans", cex=1.5)
par(mfrow=c(1,2))

# .. this parameter controls the perspective
z<-40

# .. first panel, DFE of beneficial mutations in the ancestor
after<-hist(c(x[x$back==1,]$M, x[x$back==1,]$K), breaks=30, plot=FALSE)
after$breaks<-after$breaks*0.9
after$counts<-after$counts+z

# .. plot empty frame
plot(after,col='white', xlim=c(-0.1,0.1),  ylim=c(0,500), border=NA, xlab='selection coeff. (s)', ylab='loci', cex.lab=1.5,  cex.axis=1.35, lwd=1.5, main=NA)

# .. custom x-axis
segments(-0.09,z*1.1,0.09,z*1.1, lty=2, col='grey25')

# .. custom z-axis
segments(-0.05,-0.75,-0.05*0.9,z*1.1, lty=2, col='grey25')
segments(0.05,-0.75,0.05*0.9,z*1.1, lty=2, col='grey25')
segments(-0.1,-0.75,-0.09,z*1.1, lty=2, col='grey25')
segments(0.1,-0.75,0.09,z*1.1, lty=2, col='grey25')
segments(0,-0.75,0,z*1.1, lty=2, col='grey25')

# .. plot DFE of later generations
lines(after,col=rgb(0.61,0.5,0.61,0.8), xlim=c(-0.1,0.1),  border=NA)
lines(after$breaks[1:length(after$counts)],after$counts, type='s', lwd=1.1)

# .. hide undesired section (visual trick)
rect(-0.11,0,0.1,z, border = NA, col='white')

# .. custom z-axis again
segments(-0.05,-0.75,-0.05*0.9,z*1.1, lty=2, col='grey25')
segments(0.05,-0.75,0.05*0.9,z*1.1, lty=2, col='grey25')
segments(-0.1,-0.75,-0.09,z*1.1, lty=2, col='grey25')
segments(0.1,-0.75,0.09,z*1.1, lty=2, col='grey25')
segments(0,-0.75,0,z*1.1, lty=2, col='grey25')

# .. plot DFE of beneficials in ancestor
anc<-hist(x[x$back==1,]$R, breaks=24, plot=FALSE)
lines(anc,col=rgb(0.85,0.85,0.85,0.85), xlim=c(-0.1,0.1),  border=NA)
ys<-append(0,anc$counts)
xs<-append(0.01,anc$breaks[1:length(anc$counts)])
lines(xs,ys, type='s', lwd=1.1)


# .. secnd panel, DFE of beneficial mutations in later generations
after<-hist(c(x[x$back==2,]$M,x[x$back==3,]$K), breaks=10,plot=FALSE)
after$breaks<-after$breaks*0.9
after$counts<-after$counts+z

# .. plot empty frame
plot(after,col='white', ylim=c(0,500), xlim=c(-0.1,0.1), border=NA, xlab='selection coeff. (s)', ylab='loci', cex.lab=1.5,  cex.axis=1.35, lwd=1.5, main=NA)

# .. custom x-axis
segments(-0.09,z*1.1,0.09,z*1.1, lty=2, col='grey25')

# .. custom z-axis
segments(-0.05,-0.75,-0.05*0.9,z*1.1, lty=2, col='grey25')
segments(0.05,-0.75,0.05*0.9,z*1.1, lty=2, col='grey25')
segments(-0.1,-0.75,-0.09,z*1.1, lty=2, col='grey25')
segments(0.1,-0.75,0.09,z*1.1, lty=2, col='grey25')
segments(0,-0.75,0,z*1.1, lty=2, col='grey25')

# .. plot DFE of beneficials in later generations
lines(after,col=rgb(0.61,0.5,0.61,0.8), xlim=c(-0.1,0.1),  border=NA)
ys<-append(0,after$counts)
xs<-append(0.01,after$breaks[1:length(after$counts)])
lines(xs,ys, type='s', lwd=1.1)

# .. hide undesired section (visual trick)
rect(-0.11,0,0.1,z, border = NA, col='white')

# .. custom z-axis again
segments(-0.05,-0.75,-0.05*0.9,z*1.1, lty=2, col='grey25')
segments(0.05,-0.75,0.05*0.9,z*1.1, lty=2, col='grey25')
segments(-0.1,-0.75,-0.09,z*1.1, lty=2, col='grey25')
segments(0.1,-0.75,0.09,z*1.1, lty=2, col='grey25')
segments(0,-0.75,0,z*1.1, lty=2, col='grey25')

# .. plot DFE for ancestor
anc<-hist(unique(x[x$back==2 | x$back==3,]$R), breaks=30, plot=FALSE)
lines(anc,col=rgb(0.85,0.85,0.85,0.85), xlim=c(-0.1,0.1),  border=NA)
lines(anc$breaks[1:length(anc$counts)],anc$counts, type='s', lwd=1.1)

# .. close graphical output
dev.off()
