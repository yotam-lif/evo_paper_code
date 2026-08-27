# .. This script plots the most the beneficial mutations in the ancestor and two evolved strains (2K and 15K) from the LTEE populatiion Ara+2, and shows how their effects change in the other backgrounds. From: Couce et al. (2024) Science

# .. load filtered datasets
Rtable<-read.table(file = "Rfitted_fil.txt", sep="\t", header=TRUE, as.is=TRUE, comment.char = "")
Ttable<-read.table(file = "2Kfitted_fil.txt", sep="\t", header=TRUE, as.is=TRUE, comment.char = "")
Ftable<-read.table(file = "15Kfitted_fil.txt", sep="\t", header=TRUE, as.is=TRUE, comment.char = "")

# .. remove NAs
Rtable<-Rtable[!is.na(Rtable$fitted1),]
Ttable<-Ttable[!is.na(Ttable$fitted1),]
Ftable<-Ftable[!is.na(Ftable$fitted1),]

# .. load beneficial tails for And, 2K and 15K
Rben<-Rtable[which(Rtable$fitted1<=0.3 & Rtable$fitted1>0.015 & Rtable$abn>1),]
Rben<-Rben[!duplicated(Rben$site),] 			# to impede overlapping regions being counted twice!to impede overlapping regions being counted twice!
Tben<-Ttable[which( Ttable$fitted1<=0.3 & Ttable$fitted1>0.015 & Ttable$abn>1),]
Tben<-Tben[!duplicated(Tben$site),] 			# to impede overlapping regions being counted twice!
Fben<-Ftable[which( Ftable$fitted1<=0.3 & Ftable$fitted1>0.015 & Ftable$abn>1),]
Fben<-Fben[!duplicated(Fben$site),] 			# to impede overlapping regions being counted twice!

# .. define color scheme for the plots
col2K<-'#cd6e6c'
col15K<-'#6c91cd'

# .. preparing objects for a loop to create a table with top alleles in the ancestor, 2K and 15K; and their effects in the other backgrounds
# .. first, top alleles in the ancestor
Rnames<-Rben$alle
Repi<-as.data.frame(matrix(NA, length(Rnames), 3))
n<-0
for (i in Rnames) {
	n<-n+1	
	Repi[n,1]<-Rtable[Rtable$alle%in%i,]$fitted1		# .. record fitness of the beneficial allele in the ancestor
	if (nrow(Ttable[Ttable$alle%in%i,])==0) {		# .. if the exact same allele is not present in 2K, record NA
		Repi[n,2]<-NA		
	} else {
		Repi[n,2]<-Ttable[Ttable$alle%in%i,]$fitted1		
	}	
	if (nrow(Ftable[Ftable$alle%in%i,])==0) {		# .. if the exact same allele is not present in 15K, record NA
		Repi[n,3]<-NA
	} else {
		Repi[n,3]<-Ftable[Ftable$alle%in%i,]$fitted1	
	}	
}

# .. second, top alleles in 2K
Tnames<-Tben$alle
Tepi<-as.data.frame(matrix(NA, length(Tnames), 3))
n<-0
for (i in Tnames) {
	n<-n+1		
	Tepi[n,2]<-Ttable[Ttable$alle%in%i,]$fitted1		# .. record fitness of the beneficial allele in 2K
	if (nrow(Rtable[Rtable$alle%in%i,])==0) {		# .. if the exact same allele is not present in the ancestor, record NA
		Tepi[n,1]<-NA		
	} else {
		Tepi[n,1]<-Rtable[Rtable$alle%in%i,]$fitted1		
	}	
	if (nrow(Ftable[Ftable$alle%in%i,])==0) {		# .. if the exact same allele is not present in 15K, record NA
		Tepi[n,3]<-NA
	} else {
		Tepi[n,3]<-Ftable[Ftable$alle%in%i,]$fitted1	
	}	
}

# .. finally, top alleles in 15K
Fnames<-Fben$alle
Fepi<-as.data.frame(matrix(NA, length(Tnames), 3))
n<-0
for (i in Fnames) {
	n<-n+1		
	Fepi[n,3]<-Ftable[Ftable$alle%in%i,]$fitted1		# .. record fitness of the beneficial allele in 15K
	if (nrow(Rtable[Rtable$alle%in%i,])==0) {		# .. if the exact same allele is not present in the ancestor, record NA
		Fepi[n,1]<-NA		
	} else {
		Fepi[n,1]<-Rtable[Rtable$alle%in%i,]$fitted1		
	}
	if (nrow(Ttable[Ttable$alle%in%i,])==0) {		# .. if the exact same allele is not present in 2K, record NA
		Fepi[n,2]<-NA
	} else {
		Fepi[n,2]<-Ttable[Ttable$alle%in%i,]$fitted1	
	}	
}

# .. add column names	
colnames(Repi) <- c('R', 'M', 'K')
colnames(Fepi) <- c('R', 'M', 'K')
colnames(Tepi) <- c('R', 'M', 'K')	

# .. initialize graphical output
png(file="segben.png", width = (15.63/2)*0.85, height = 8.54/1.9, units = 'in', res = 300)
par(family="sans", cex=2)
par(mfrow=c(1,2))
par(lwd = 1.5)

# .. first panel, plot points corresponding to fitness values of the top beneficial alleles in the ancestor
plot(rep(1,nrow(Repi)), Repi$R, xlim=c(0.8,2.2), ylim=c(-0.15,0.1), cex=1, lwd=0.5, ylab='selection coeff. (s)', cex.lab=1.5,  cex.axis=1.35)

# .. add reference line
abline(h=0, lty=2)

# .. add arrows connecting the above points to the corresponding fitness values in the 2K background 
arrows(1,Repi$R[1:nrow(Repi)],2,Repi$M[1:nrow(Repi)], col= gray (0, .5), lwd=0.75, length=0.075, angle=25)

# .. plot points corresponding to fitness values of the top beneficial alleles in the 2K background 
points(rep(2,nrow(Tepi)),Tepi$M, col=col2K, lwd=0.5, cex=1)

# .. add arrows connecting the above points to the corresponding fitness values in the ancestral background 
arrows(2,Tepi$M[1:nrow(Tepi)],1,Tepi$R[1:nrow(Tepi)], col= rgb(0.8,0.43,0.42, .5), lwd=0.75, length=0.075, angle=25)

# .. new panel, plot points corresponding to fitness values of the top beneficial alleles in the 2K background
plot(rep(1,nrow(Tepi)),Tepi$M, xlim=c(0.8,2.2), ylim=c(-0.15,0.1), col=col2K, cex=1, lwd=0.5, ylab='selection coeff. (s)', cex.lab=1.5,  cex.axis=1.35)

# .. add reference line
abline(h=0, lty=2)

# .. add arrows connecting the above points to the corresponding fitness values in the 15K background 
arrows(1,Tepi$M[1:nrow(Tepi)],2,Tepi$K[1:nrow(Tepi)], col= rgb(0.8,0.43,0.42, .5), lwd=0.75, length=0.075, angle=25)

# .. plot points corresponding to fitness values of the top beneficial alleles in the 15K background 
points(rep(2,nrow(Fepi)),Fepi$K, col=rgb(0.42,0.57,0.8, .5), lwd=0.5, cex=1)

# .. add arrows connecting the above points to the corresponding fitness values in the 2K background 
arrows(2,Fepi$K[1:nrow(Fepi)],1,Fepi$M[1:nrow(Fepi)], col=rgb(0.42,0.57,0.8, .5), lwd=0.75, length=0.075, angle=25)

# .. close graphical output
dev.off()
