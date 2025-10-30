<Cabbage> bounds(0, 0, 0, 0)
form caption("Snare2") size(700, 404), guiMode("queue") pluginId("rb39")

button  bounds(234, 2, 71, 40) channel("trig") text("TrigAll") textColour("white")
button  bounds(332, 2, 71, 40) channel("stop") text("StopAll") textColour("white")

hslider bounds(170, 44, 313, 32) channel("vol") range(0.0, 1, .7, 0.5, .01) text("MasterVolume") textColour("white")
hslider bounds(170, 76, 313, 32) channel("dur") range(1, 2000, 1000, 0.5, .01) text("MasterDuration") textColour("white")
hslider bounds(172, 110, 313, 32) channel("rate") range(.9, 1.7, 1, 0.5, .01) text("MasterRate") textColour("white")
hslider bounds(172, 142, 313, 32) channel("verb") range(0, 0.6, .5, 0.5, .01) text("MasterReverb") textColour("white")

hslider bounds(334, 178, 313, 32) channel("trans1") range(0.0001, 1, .2, 0.5, 0.01) text("Transpose1") textColour("white")
hslider bounds(334, 212, 313, 32) channel("trans2") range(0.0001, 1, .1, 0.5, 0.01) text("Transpose2") textColour("white")
hslider bounds(334, 240, 313, 32) channel("trans3") range(0.001,  3, .1, 0.5, 0.01) text("Transpose3") textColour("white")
hslider bounds(334, 270, 313, 32) channel("trans4") range(0.001,  4, .3, 0.5, 0.01) text("Transpose4") textColour("white")
hslider bounds(334, 300, 313, 32) channel("trans5") range(0.01,  7, 1, 0.5, 0.01) text("Transpose5") textColour("white")

hslider bounds(16, 178, 313, 32) channel("vol1") range(0.0, 3, 2, 0.5, 0.01) text("Volume1") textColour("white")
hslider bounds(16, 210, 313, 32) channel("vol2") range(0.0, 3, 2.5, 0.5, 0.01) text("Volume2") textColour("white")
hslider bounds(16, 240, 313, 32) channel("vol3") range(0.0, 4, 3.6, 0.5, 0.01) text("Volume3") textColour("white")
hslider bounds(16, 270, 313, 32) channel("vol4") range(0.0, 5, 3.9, 0.5, 0.01) text("Volume4") textColour("white")
hslider bounds(18, 302, 313, 32) channel("vol5") range(0.0, 6, 4.6, 0.5, 0.01) text("Volume5") textColour("white")

combobox bounds(264, 370, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") 
filebutton bounds(266, 340, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(330, 340, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
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

chn_a "reverbsendl", 3
chn_a "reverbsendr", 3

seed 0

    instr Trigger
kTrigAll chnget "trig"
  if changed(kTrigAll) == 1 then
    event "i", 111,  0.0, chnget:i("dur")
  endif
  endin
    
    instr Stop
kStopAll chnget "stop"
  if changed(kStopAll) == 1 then
    turnoff2 111,   0.0, 0
    turnoff2 116,   0.0, 0        
    turnoff2 117,   0.0, 0
    turnoff2 118,   0.0, 0 
    turnoff2 119,   0.0, 0
    turnoff2 120,   0.0, 0        
  endif    
        endin


  instr 111 ; trigger drum hits
krate       init 333
ktrigger    metrobpm    krate * chnget:k("rate")                   ; rate of drum strikes
kdrum       random      116, 120.999                               ; randomly choose drum to strike
            schedkwhen  ktrigger, 0, 0, kdrum, 0, 0.1              ; strike a drum
  endin

  instr	116	; drum 1 					
p3		=		0.01								   ; DURATION FOR THIS SOUND
aenv expon	0.1+rnd(0.3),p3+rnd(.2),0.001	           ; AMPLITUDE ENVELOPE - PERCUSSIVE DECAY
asig noise	aenv+rnd(.2), 0					           ; CREATE SOUND
asig buthp	asig, (1000+rnd(5000))*chnget:k("trans1")  ; HIGHPASS FILTER SOUND
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 asig, kpan
aL = aL*chnget:k("vol")*chnget:k("vol1")
aR = aR*chnget:k("vol")*chnget:k("vol1")
outs aL, aR
chnmix aL*rnd(.52)*chnget:k("verb"), "reverbsendl"
chnmix aR*rnd(.41)*chnget:k("verb"), "reverbsendr"
    endin

    instr	117	; drum 2
p3	=	.3										            ; DURATION FOR THIS SOUND
aenv expon	0.1+rnd(.3),p3+rnd(.5),0.001		            ; AMPLITUDE ENVELOPE - PERCUSSIVE DECAY
asig noise	aenv, 0								            ; CREATE SOUND 
asig buthp	asig, (2000+rnd(3000))*chnget:k("trans2")		; HIGHPASS FILTER SOUND	
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 asig, kpan
aL = aL*chnget:k("vol")*chnget:k("vol2")
aR = aR*chnget:k("vol")*chnget:k("vol2")
outs aL, aR
chnmix aL*rnd(.62)*chnget:k("verb"), "reverbsendl"
chnmix aR*rnd(.41)*chnget:k("verb"), "reverbsendr"
    endin

    instr	118	; drum 3
p3		=		.6										; DURATION FOR THIS SOUND
asig tambourine .1+rnd(.4), p3+rnd(.5), 3000+rnd(7000), 0.1+rnd(0.5), 0, 10+rnd(20)*chnget:i("trans3"),50+rnd(100)*chnget:i("trans3"),80+rnd(100)*chnget:i("trans3")
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 asig, kpan
aL = aL*chnget:k("vol")*chnget:k("vol3")
aR = aR*chnget:k("vol")*chnget:k("vol3")
outs aL, aR
chnmix aL*rnd(.22)*chnget:k("verb"), "reverbsendl"
chnmix aR*rnd(.31)*chnget:k("verb"), "reverbsendr"
    endin

    instr	119	; drum 4
p3		=		.7										; DURATION FOR THIS SOUND
asig tambourine .1+rnd(.4), p3+rnd(.4), 2000+rnd(6000), 0.1+rnd(0.5), 0, 40+rnd(50)*chnget:i("trans4"), 70+rnd(200)*chnget:i("trans4"), 90+rnd(300)*chnget:i("trans4")
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 asig, kpan
aL = aL*chnget:k("vol")*chnget:k("vol4")
aR = aR*chnget:k("vol")*chnget:k("vol4")
outs aL, aR
chnmix aL*rnd(.52)*chnget:k("verb"), "reverbsendl"
chnmix aR*rnd(.41)*chnget:k("verb"), "reverbsendr"
    endin

    instr	120	; drum 5
p3		=		.7										; DURATION FOR THIS SOUND
asig tambourine .1+rnd(.4),p3+rnd(.3),1000+rnd(5000),0.1+rnd(0.5),0,(80+rnd(50))*chnget:i("trans5"), (130+rnd(200))*chnget:i("trans5"), (390+rnd(300))*chnget:i("trans5")
kpan rspline -1, 1, 0.1, 2
aL,aR pan2 asig*0.6, kpan
aL = aL * chnget:k("vol")*chnget:k("vol5")
aR = aR * chnget:k("vol")*chnget:k("vol5")
outs aL, aR
chnmix aL * rnd(.42) * chnget:k("verb"), "reverbsendl"
chnmix aR * rnd(.51) * chnget:k("verb"), "reverbsendr"
 endin

	instr Reverb
arevl chnget "reverbsendl"
arevr chnget "reverbsendr"
denorm(arevl,arevr)
kfb init .8
kcf init 8000
al,ar reverbsc arevl, arevr, .98, 11000
outs al, ar
chnclear "reverbsendl","reverbsendr"
	endin

</CsInstruments>
<CsScore>
i "Trigger" 0 [60*60*24*7] 
i "Stop"    0 [60*60*24*7] 
i "Reverb"  0 [60*60*24*7] 
</CsScore>
</CsoundSynthesizer>