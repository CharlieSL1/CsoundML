<Cabbage> bounds(0, 0, 0, 0)
form caption("Drum1") size(929, 455), guiMode("queue") pluginId("rb37")

button  bounds(370, 8, 71, 40) channel("trigAll") text("TrigAll") textColour("white")
button  bounds(468, 8, 71, 40) channel("stopAll") text("StopAll") textColour("white")

hslider bounds(304, 78, 320, 32) channel("volAll") range(0.0, 1, .7, 0.5, .01) text("MasterVolume") textColour("white")
hslider bounds(302, 50, 321, 29) channel("durAll") range(1, 180, 30, 0.5, .01) text("MasterDuration") textColour("white")

hslider bounds(304, 108, 323, 30) channel("rateAll") range(.1, 100, 3, 0.5, .01) text("MasterRate") textColour("white")
hslider bounds(46, 222, 299, 30) channel("rateKick") range(.1, 100, 3, 0.5, .01) text("KickRate") textColour("white")
hslider bounds(602, 212, 313, 30) channel("rateStick") range(.1, 100, 3, 0.5, .01) text("StickRate") textColour("white")
hslider bounds(316, 322, 314, 30) channel("rateSnare") range(.1, 100, 3, 0.5, .01) text("SnareRate") textColour("white")

button bounds(130, 110, 72, 40) channel("trigKick") text("TrigKick") textColour("white")
button bounds(204, 110, 71, 40) channel("stopKick") text("StopKick") textColour("white")
button bounds(686, 108, 75, 37) channel("trigStick") text("TrigStick") textColour("white")
button bounds(764, 108, 63, 38) channel("stopStick") text("StopStick") textColour("white")
button bounds(390, 212, 75, 37) channel("trigSnare") text("TrigSnare") textColour("white")
button bounds(470, 212, 63, 38) channel("stopSnare") text("StopSnare") textColour("white")

hslider bounds(50, 154, 295, 33) channel("volKick") range(0.1, 5, 2,.5,.01) text("KickVolume") textColour("white")
hslider bounds(604, 150, 312, 33) channel("volStick") range(0.1, 4, 2,.5,.01) text("StickVolume") textColour("white")
hslider bounds(316, 256, 312, 33) channel("volSnare") range(0.1, 4, 2,.5,.01) text("SnareVolume") textColour("white")

hslider bounds(48, 192, 298, 32) channel("transKick") range(0.4, 4, 1, 0.5, 0.01) text("KickTranspose") textColour("white")
hslider bounds(602, 182, 312, 31) channel("transStick") range(0.25, 60, 1, 0.5, 0.01) text("StickTranspose") textColour("white")
hslider bounds(316, 292, 312, 31) channel("transSnare") range(0.5, 50, 1, 0.5, 0.01) text("SnareTranspose") textColour("white")

combobox bounds(396, 396, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") 
filebutton bounds(398, 366, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(462, 366, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
</Cabbage>
<CsoundSynthesizer>
<CsOptions>
-dm0
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

; inspired by and Modelled on Iain McCurdy's Generative Drum Machine (March 2024)

giSine       ftgen       0, 0, 2^16, 10, 1 ;a sine wave

    instr Trigger
kTrigAll chnget "trigAll"
  if changed(kTrigAll) == 1 then
    event "i",  1,  0.0, chnget:i("durAll")
    event "i", 11,  0.1, chnget:i("durAll") 
    event "i", 22,  0.2, chnget:i("durAll")      
    event "i", 33,  0.3, chnget:i("durAll")
  endif
kTrigKick chnget "trigKick"
  if changed(kTrigKick) == 1 then
    event "i", 1,  0.0, chnget:i("durAll")
  endif
kTrigStick chnget "trigStick"
  if changed(kTrigStick) == 1 then
    event "i", 11,  0.0, chnget:i("durAll")
  endif
kTrigSnare chnget "trigSnare"
  if changed(kTrigSnare) == 1 then
    event "i", 22,  0.0, chnget:i("durAll")
  endif
  endin
    
    instr Stop
kStopKick chnget "stopKick"
  if changed(kStopKick) == 1 then
    turnoff2 1,   0.0, 0
    turnoff2 2,   0.0, 0        
   endif
kStopStick chnget "stopStick"
  if changed(kStopKick) == 1 then
    turnoff2 11,   0.0, 0
    turnoff2 12,   0.0, 0 
  endif       
kStopSnare chnget "stopSnare"
  if changed(kStopSnare) == 1 then
    turnoff2 22,   0.0, 0
    turnoff2 23,   0.0, 0        
  endif
        endin


  instr 1 ;trigger lo drum hits
ktrigger    metro    chnget:k("rateAll") * chnget:k("rateKick") ; rate of drum strikes
kdrum       random      2, 5.999; randomly choose drum to strike
            schedkwhen  ktrigger, 0, 0, kdrum, 0, 0.1; strike a drum
  endin
  
  instr 2 ; sound 1 - low drum
iamp        random      0, 1; amplitude randomly chosen from between the given values
p3          =           0.956; define duration for this sound
aenv        expon       1, p3, 0.001; amplitude envelope - percussive decay
icps        exprand     33+rnd(3); cycles-per-second offset randomly chosen from an exponential distribution
kcps        expon       icps, p3, 33; pitch glissando
aSig        oscil       aenv*iamp, kcps*chnget:k("transKick"), giSine
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll") * chnget:k("volKick")
aR = aR * chnget:k("volAll") * chnget:k("volKick")
outs aL, aR
  endin 

instr 3
endin
instr 4
endin
instr 5
endin

  instr 11 ; trigger stick drum hits
ktrigger    metro    chnget:k("rateAll") * chnget:k("rateStick") ; rate of drum strikes
kdrum       random      12, 14.999; randomly choose drum to strike
            schedkwhen  ktrigger, 0, 0, kdrum, 0, 0.1; strike a drum
  endin
 
  instr 12 ; sound  - stick drum - rimshot
iamp        random      0, 0.8 ; amplitude randomly chosen from between the given values
p3          =           0.02  ; define duration for this sound
aenv        expon       1, p3, 0.001 ; amplitude envelope - percussive decay
icps        exprand     600+rnd(100) ; cycles-per-second offset randomly chosen from an exponential distribution
kcps        expon       icps, p3, 600 ; pitch glissando
aSig        poscil      aenv*iamp, kcps*chnget:k("transStick"), giSine
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll") * chnget:k("volStick")
aR = aR * chnget:k("volAll") * chnget:k("volStick")
outs aL, aR            
  endin

  instr 13
  endin  
  instr 14
  endin


  instr 22 ; trigger - hi drum hits
imet        =           1
imov        =           p3/5
krate       linseg      imet,imov,imet*8,imov,imet*2,imov,imet*7,imov,imet*3,imov,imet*.9,imov,imet*.1
ktrigger    metro       krate * chnget:k("rateSnare") * .12 ; rate of drum strikes
kdrum       random      22, 28.999 ; randomly choose drum to strike
            schedkwhen  ktrigger, 0, 0, kdrum, 0, 0.1 ; strike a drum
  endin
 
  instr 23 ; sound - hi drum
iamp        random      0, 0.4; amplitude randomly chosen from between the given values
p3          =           0.01 ; define duration for this sound
aenv        expon       1,p3,0.001 ; amplitude envelope - percussive decay
icps        exprand     3200+rnd(500); cycles-per-second offset randomly chosen from an exponential distribution
kcps        expon       icps+2000,p3, 100 ; pitch glissando
aSig        oscil       aenv*iamp, kcps*chnget:k("transSnare"), giSine
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll") * chnget:k("volSnare")
aR = aR * chnget:k("volAll") * chnget:k("volSnare")
outs aL, aR   
endin
  
  instr 24 
  endin  
  instr 25 
  endin  
  instr 26
  endin
  instr 27
  endin
  instr 28
  endin
  
  instr 33 ;trigger hats and snare hits
ktrigger    metro    chnget:k("rateAll") * chnget:k("snareRate") ; rate of drum strikes
kdrum       random      33, 48.999; randomly choose drum to strike
            schedkwhen  ktrigger, 0, 0, kdrum, 0, 0.1 ; strike a drum
  endin

instr	34	;SOUND 4 - CLOSED HI-HAT					
	p3		=		0.01								   ;DEFINE DURATION FOR THIS SOUND
	aenv		expon	0.1+rnd(0.3),p3+rnd(.2),0.001	   ;AMPLITUDE ENVELOPE - PERCUSSIVE DECAY
	aSig		noise	aenv+rnd(.2), 0					   ;CREATE SOUND FOR CLOSED HI-HAT
	aSig		buthp	aSig, 1000+rnd(5000)			   ;HIGHPASS FILTER SOUND

kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan

aL = aL * chnget:k("volAll")
aR = aR * chnget:k("volAll")
 
outs aL, aR   

endin

  instr 35
  endin
  instr 36
  endin    

    instr	37	;SOUND 5 - OPEN HI-HAT
p3		=		.3										   ;DEFINE DURATION FOR THIS SOUND
aenv		expon	0.1+rnd(.3),p3+rnd(.5),0.001		;AMPLITUDE ENVELOPE - PERCUSSIVE DECAY
aSig		noise	aenv, 0								;CREATE SOUND FOR CLOSED HI-HAT
aSig		buthp	aSig, 2000+rnd(3000)			   ;HIGHPASS FILTER SOUND	
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll")
aR = aR * chnget:k("volAll")
outs aL, aR   
    endin

  instr 38
  endin  
  instr 39
  endin 
   
    instr	40	;SOUND 6 - snare-ish
p3		=		.6										;DEFINE DURATION FOR THIS SOUND
aSig	tambourine .1+rnd(.4), p3+rnd(.35), 3000+rnd(7000), 0.1+rnd(0.5), 0, 2000+rnd(2000), 3000+rnd(3000), 4000+rnd(4000)
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll")
aR = aR * chnget:k("volAll")
outs aL, aR   
    endin

  instr 41
  endin  
  instr 42
  endin 
  instr 43
  endin 

    instr	44	;SOUND 7
	p3		=		.5										;DEFINE DURATION FOR THIS SOUND
	aSig		tambourine .1+rnd(.4), p3+rnd(.4), 2000+rnd(6000), 0.1+rnd(0.5), 0, 4000+rnd(400), 700+rnd(2000), 900+rnd(3000)
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll")
aR = aR * chnget:k("volAll")
outs aL, aR   
endin

      instr 45
      endin    
      instr 46
      endin  
      instr 47
      endin

        instr	48	;SOUND 8
	p3		=		.4										;DEFINE DURATION FOR THIS SOUND
	aSig		tambourine .1+rnd(.4), p3+rnd(.3), 1000+rnd(8000), 0.1+rnd(0.5), 0, 8000+rnd(50), 1390+rnd(2000), 3900+rnd(3000)

kpan rspline -1, 1, 0.1, 2
aL,aR pan2 aSig, kpan
aL = aL * chnget:k("volAll")
aR = aR * chnget:k("volAll")
outs aL, aR   
endin
</CsInstruments>
<CsScore>
i "Trigger" 0 [60*60*24*7] 
i "Stop"    0 [60*60*24*7] 
/*
i1    00 181 400 0.9 ; start low drum trigger instr
i11   10  16 220 0.7 ; start mid drum trigger instr
i21   20  20 102 0.6 ; start  hi drum trigger instr
i111  30  15 426 0.21
i11   42  16 210 0.7
i11   49  20 218 0.7
i21   54  30 130 0.7
i111  60  20 479 0.2
i11   66  20 206 0.9
i1    79  35 100 0.6
i111  93  20 512 0.21
i11   96  30 111 0.6
i21  110  15 140 0.6
i111 120  30 412 0.21
i111 123  30 420 0.21
i1   127  30 500 0.5
i11  128  16 209 0.3
i1   130  50 190 0.7
i11  133  16 210 0.6 
i21  140  20 201 0.5 
i11  153  10 210 0.6 
i21  161  10 201 0.5
*/
</CsScore>
</CsoundSynthesizer>