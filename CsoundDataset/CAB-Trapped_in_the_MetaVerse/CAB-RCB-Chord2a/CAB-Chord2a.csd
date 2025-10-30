<Cabbage> bounds(0, 0, 0, 0)
form caption("Chords") size(666, 366), guiMode("queue") pluginId("rb34")

button  bounds(44, 6, 71, 40)   channel("trgC") text("Trigger Chords") textColour("white")
button  bounds(182, 6, 71, 40)   channel("stopC") text("Stop Chords") textColour("white")
button  bounds(384, 4, 61, 47)   channel("trgG") text("Trigger Glitch") textColour("white")
button  bounds(520, 2, 63, 52)   channel("stopG") text("Stop Glitch") textColour("white")

hslider bounds(16, 122, 295, 34) channel("durC") range(1, 180, 30,.5,.01) text("ChordDuration") textColour("white")
hslider bounds(326, 120, 310, 37) channel("durG") range(2, 6, 3,.5,.01) text("GlitchDuration") textColour("white")

hslider bounds(320, 224, 317, 30) channel("rate") range(1, 1000, 60, 0.5,.01) text("GlitchRate") textColour("white")

hslider bounds(14, 88, 298, 32)  channel("trnsC") range(0.1, 6, 1.0,.5,.01)   text("ChordTranspose") textColour("white")
hslider bounds(326, 88, 312, 31)  channel("trnsG") range(0.5, 50, 1.0,.5,.01)   text("GlitchTranspose") textColour("white")

hslider bounds(16, 54, 295, 33)  channel("volC") range(0.0, 1, 0.5,.5,.01)   text("ChordVolume")    textColour("white")
hslider bounds(326, 54, 312, 33)  channel("volG") range(0.0, 1, 0.5,.5,.01)   text("GlitchVolume")    textColour("white")

hslider bounds(14, 192, 294, 29) channel("delC") range(0.0, 1, 0.4,.5,.01)   text("ChordDelay")     textColour("white")
hslider bounds(322, 190, 315, 29) channel("delG") range(0.0, 1, 0.4,.5,.01)   text("GlitchDelay")     textColour("white")
hslider bounds(16, 160, 295, 29) channel("revC") range(0.0, 1, 0.7,.5,.01)   text("ChordReverb")    textColour("white")
hslider bounds(324, 158, 313, 29) channel("revG") range(0.0, 1, 0.7,.5,.01)   text("GlitchReverb")    textColour("white")

combobox bounds(230, 306, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") text("LowWithGlitch", "HighwithGlitch", "MostlyGlitch", "MidChordsVerb")
filebutton bounds(232, 276, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(296, 276, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
</Cabbage>

<CsoundSynthesizer>
<CsOptions>
-d -n
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

chn_a "reverbsendl", 3
chn_a "reverbsendr", 3
chn_a "delaysend1", 3
chn_a "delaysend2", 3

seed 0

    instr Trigger
kTrigC chnget "trgC"
  if changed(kTrigC) == 1 then
    event "i", 1,  0.0, chnget:i("durC")
    event "i", 2,  0.1, chnget:i("durC")
    event "i", 3,  0.2, chnget:i("durC") 
    event "i", 4,  0.3, chnget:i("durC") 
  endif
 kTrigG chnget "trgG"
  if changed(kTrigG) == 1 then
    event "i", 5,  0.0, chnget:i("durG")
    event "i", 6,  0.1, chnget:i("durG") 
  endif
    endin
    
    instr Stop
kStopC chnget "stopC"
  if changed(kStopC) == 1 then
    turnoff2 1,   0.0, 0
    turnoff2 10,  0.0, 0    
    turnoff2 2,   0.1, 0
    turnoff2 20,  0.1, 0    
    turnoff2 3,   0.2, 0
    turnoff2 30,  0.2, 0
    turnoff2 4,   0.3, 0
    turnoff2 40,  0.3, 0     
   endif
kStopG chnget "stopG"
  if changed(kStopC) == 1 then
    turnoff2 5,   0.0, 0
    turnoff2 50,  0.0, 0    
    turnoff2 6,   0.1, 0
    turnoff2 60,  0.1, 0        
   endif
    endin

	instr 1 
gktempo1 init 76
gktick1  metrobpm gktempo1
gkschedrnd1 trandom gktick1, 1, 5
	if (int(gkschedrnd1) == 1) then
		gksched1 = 1
   		else gksched1 = 0
	endif
gkdurrnd1 trandom gktick1, 3, 7
schedkwhen gksched1, 0, 1, 10, 0, 1+gkdurrnd1+9
schedkwhen gksched1, 0, 1, 10, 0, 1+gkdurrnd1+9
schedkwhen gksched1, 0, 1, 10, 0, 1+gkdurrnd1+9
	endin

	instr 10 	
giNotes[] fillarray 12, 12, 16, 16, 7, 19, 0, 9 
gioffset[] fillarray 12, 0, 24 
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsC")*cpsmidinn(knote+24+ioffset)
kmodf = kcps*2
idur random 5,12
aenv linseg 0, 0.3*idur, 1, 0.24*idur, 1, 0.26*idur, 0
aindx linseg 0, 0.1, 1, 0.1*idur+rnd(0.2), 4, 0.44*idur, 2, 0.05*idur, 1
amod oscili aindx*kmodf, kmodf+rnd(1), -1, 0.25
acar oscili 1, (kcps+amod)
kbreathcf rspline 2000, 7000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
aout = (acar+(abreath*rnd(0.34)))*iamp*aenv
aout butterlp aout, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 aout, kpan
outs aL*chnget:k("volC"), aR*chnget:k("volC")
chnmix aL*chnget:k("delC"), "delaysend1"
chnmix aR*chnget:k("delC"), "delaysend2"
chnmix aL*chnget:k("revC"), "reverbsendl"
chnmix aR*chnget:k("revC"), "reverbsendr"
	endin

	instr 2 
gktempo2 init 77
gktick2  metrobpm gktempo2
gkschedrnd2 trandom gktick2, 1, 8
	if (int(gkschedrnd2) == 1) then
		gksched2 = 1
   		else gksched2 = 0
	endif
gkdurrnd2 trandom gktick2, 4, 11
schedkwhen gksched2, 0, 1, 20, 0, gkdurrnd2+7
schedkwhen gksched2, 0, 1, 20, 0, gkdurrnd2+9
schedkwhen gksched2, 0, 1, 20, 0, gkdurrnd2+11
    endin

	instr 20
giNotes[] fillarray 12, 14, 0, 0, 12, 7, 0, 22 
gioffset[] fillarray -12, 0, 12 	
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsC")*(cpsmidinn(knote+36+ioffset))
kmodf = kcps
idur random 4,12
aenv linseg 0, 0.3*idur, 1, 0.24*idur, .6, 0.26*idur, 0
aindx linseg 0, 0.1, 4, 0.1*idur+rnd(0.2), 6, 0.44*idur, 2, 0.05*idur, 2
amod oscili aindx*kmodf, kmodf+rnd(1), -1, 0.25
acar oscili 1, (kcps+amod)
kbreathcf rspline 3000, 6000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
aout = (acar+(abreath*rnd(1.01)))*iamp*aenv
aout butterlp aout, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 aout, kpan
outs aL*chnget:k("volC"), aR*chnget:k("volC")
chnmix aL*chnget:k("delC"), "delaysend1"
chnmix aR*chnget:k("delC"), "delaysend2"
chnmix aL*chnget:k("revC"), "reverbsendl"
chnmix aR*chnget:k("revC"), "reverbsendr"
	endin

	instr 3 
gktempo3 init 70
gktick3  metrobpm gktempo3
gkschedrnd3 trandom gktick3, 1, 4
	if (int(gkschedrnd3) == 1) then
		gksched3 = 1
   		else gksched3 = 0
	endif
gkdurrnd3 trandom gktick3, 4, 11
schedkwhen gksched3, 0, 1, 30, 0, gkdurrnd3+4
schedkwhen gksched3, 0, 1, 30, 0, gkdurrnd3+6
schedkwhen gksched3, 0, 1, 30, 0, gkdurrnd3+9
endin

	instr 30 
giNotes[] fillarray 12, 7, 0, 14, 22, 18, 24, 20
gioffset[] fillarray -12, 0, 12 	
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsC")*cpsmidinn(knote+48+ioffset)
kmodf = kcps*.5
idur random 3,13
aenv linseg 0, 0.3*idur, 1, 0.24*idur, 1, 0.26*idur, 0
aindx linseg 0, 0.1, 7, 0.1*idur+rnd(0.2), 2, 0.44*idur, 9, 0.05*idur, 2
amod oscili aindx*kmodf, kmodf, -1, 0.25
acar oscili 1, (kcps+amod)
kbreathcf rspline 1000, 8000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
asig = acar+abreath*rnd(0.4)
asig butterlp asig, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 asig*iamp*aenv, kpan
outs aL*chnget:k("volC"), aR*chnget:k("volC")
chnmix aL*chnget:k("delC"), "delaysend1"
chnmix aR*chnget:k("delC"), "delaysend2"
chnmix aL*chnget:k("revC"), "reverbsendl"
chnmix aR*chnget:k("revC"), "reverbsendr"
	endin

	instr 4 
gktempo4 init 90
gktick4  metrobpm gktempo4
gkschedrnd4 trandom gktick4, 1, 3
	if (int(gkschedrnd4) == 1) then
		gksched4 = 1
   		else gksched4 = 0
	endif
gkdurrnd4 trandom gktick4, 2, 5
schedkwhen gksched4, 0, 1, 40, 0, gkdurrnd4+3
schedkwhen gksched4, 0, 1, 40, 0, gkdurrnd4+5
schedkwhen gksched4, 0, 1, 40, 0, gkdurrnd4+8
endin

	instr 40 
giNotes[] fillarray 0, 4, 0, 7, 0, 5, 0, 9
gioffset[] fillarray 12, 24, 0 	
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsC")*(cpsmidinn(knote+36+(rnd(12))+ioffset))
kmodf = kcps*2.02
idur = p3
aenv linseg 0, 0.3*idur, 1, 0.24*idur, 1, 0.26*idur, 0
aindx linseg 0, 0.1, 2, 0.1*idur+rnd(0.2), 5, 0.44*idur, 1, 0.05*idur, 2
amod oscili aindx*kmodf, kmodf, -1, 0.25
acar oscili 1, (kcps+amod)
kbreathcf rspline 1000, 8000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
aout = (acar+(abreath*rnd(0.4)))*iamp*aenv
aout butterlp aout, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 aout, kpan
outs aL*chnget:k("volC"), aR*chnget:k("volC")
chnmix aL*chnget:k("delC"), "delaysend1"
chnmix aR*chnget:k("delC"), "delaysend2"
chnmix aL*chnget:k("revC"), "reverbsendl"
chnmix aR*chnget:k("revC"), "reverbsendr"
	endin
	
	instr 5 
gktick5 init 60
gktick5 metro chnget:k("rate")
gkschedrnd5 trandom gktick5, 1, 3
	if (int(gkschedrnd5) == 1) then
		gksched5 = 1
   		else gksched5 = 0
	endif
gkdurrnd5 trandom gktick5, .05, .1
schedkwhen gksched5, 0, 1, 50, 0, gkdurrnd5
    endin

	instr 50 
giNotes[] fillarray 10, 17, 20, 27, 30, 24, 11, 34
gioffset[] fillarray 12, 0, -12 	
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsG")*cpsmidinn(knote+72+ioffset)
kmodf = kcps*1+rnd(20)
idur random .01, .25
aenv linseg 0, 0.3*idur, 1, 0.24*idur, 1, 0.26*idur, 0
aindx linseg 0, 0.1, 8, 0.1*idur+rnd(0.2), 2, 0.44*idur, 1, 0.05*idur, 2
amod oscili aindx*kmodf, kmodf+rnd(1), -1, 0.25
acar oscili 1, (kcps+amod)
kbreathcf rspline 1000, 8000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
aout = (acar+(abreath*rnd(0.74)))*iamp*aenv
aout butterlp aout, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 aout, kpan
outs aL*chnget:k("volG"), aR*chnget:k("volG")
chnmix aL*chnget:k("delG"), "delaysend1"
chnmix aR*chnget:k("delG"), "delaysend2"
chnmix aL*chnget:k("revG"), "reverbsendl"
chnmix aR*chnget:k("revG"), "reverbsendr"
	endin

	instr 6 
gktick6 init 2
gktick6  metro chnget:k("rate")
gkschedrnd6 trandom gktick6, 1, 3
	if (int(gkschedrnd6) == 1) then
		gksched6 = 1
   		else gksched6 = 0
	endif
gkdurrnd6 trandom gktick6, .01, .04
schedkwhen gksched6, 0, 1, 60, 0, gkdurrnd6
    endin

	instr 60 
giNotes[] fillarray 10, 17, 20, 27, 30, 24, 11, 36
gioffset[] fillarray 12, 0, 24 	
ioffind random 0, 3
ioffind = int(ioffind)
ioffset = gioffset[ioffind]
iindex random 0, 8
iindex = int(iindex)
knote = giNotes[iindex]
iamp = 0.15
kcps = chnget:k("trnsG")*cpsmidinn(knote+48+ioffset)
kmodf = kcps*1+rnd(21)
idur random .01,.2
aenv linseg 0, 0.3*idur, 1, 0.24*idur, 1, 0.26*idur, 0
aindx linseg 0, 0.1, 8, 0.1*idur+rnd(0.2), 2, 0.44*idur, 10, 0.05*idur, 2
amod oscili aindx*kmodf, kmodf+rnd(10), -1, 0.25
acar oscili 1, kcps+amod
kbreathcf rspline 1000, 8000, 0.1, 4
abreath pinker
abreath butterlp abreath, kbreathcf
abreath butterhp abreath, 1000
aout = (acar+(abreath*rnd(0.74)))*iamp*aenv
aout butterlp aout, 7000+rnd(1000)
kpan rspline -1, 1, 0.5, 4
aL, aR pan2 aout, kpan
outs aL*chnget:k("volG"), aR*chnget:k("volG")
chnmix aL*chnget:k("delG"), "delaysend1"
chnmix aR*chnget:k("delG"), "delaysend2"
chnmix aL*chnget:k("revG"), "reverbsendl"
chnmix aR*chnget:k("revG"), "reverbsendr"
	endin
	
    instr DelayVerb1
adel chnget "delaysend1"
khpf init 300
klpf init 8000
klpfmod1 lfo 1000, 4
klpfmod2 lfo 1000, 5
kfb randi .5, 2
irates[] fillarray 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
kdelayrate = 8
afb init 0
adel butterhp adel, khpf
kdelay = ((gkdurrnd5*irates[kdelayrate]))*4*100
adelayl vdelay adel+afb, kdelay, 500
adelayr vdelay adel+afb, kdelay*0.75, 500
adelayl butterlp adelayl, klpf+klpfmod1
adelayr butterlp adelayr, klpf+klpfmod2
afb = ((adelayl + adelayr)*0.5) * kfb
aoutl, aoutr freeverb adelayl, adelayr, 0.64, 0.3
al ntrpol adelayl, aoutl, 0.5
ar ntrpol adelayr, aoutr, 0.5
outs al*chnget:k("volC")*chnget:k("volG"), ar*chnget:k("volC")*chnget:k("volG")
chnclear "delaysend1"
    endin

    instr DelayVerb2
adel chnget "delaysend2"
khpf init 300
klpf init 8000
klpfmod1 lfo 1000, 4
klpfmod2 lfo 1000, 5
kfb randi .5, 2
irates[] fillarray 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
kdelayrate = 8
afb init 0
adel butterhp adel, khpf
kdelay = ((gkdurrnd5*irates[kdelayrate]))*4*100
adelayl vdelay adel+afb, kdelay, 500
adelayr vdelay adel+afb, kdelay*0.75, 500
adelayl butterlp adelayl, klpf+klpfmod1
adelayr butterlp adelayr, klpf+klpfmod2
afb = ((adelayl + adelayr)*0.5) * kfb
aoutl, aoutr freeverb adelayl, adelayr, 0.64, 0.3
al ntrpol adelayl, aoutl, 0.5
ar ntrpol adelayr, aoutr, 0.5
outs al*chnget:k("volG"), ar*chnget:k("volG")
chnclear "delaysend2"
    endin

	instr ReverbC
arevl chnget "reverbsendl"
arevr chnget "reverbsendr"
kfb init 0.98
kcf init 8000
al, ar reverbsc arevl, arevr, kfb, kcf
outs al*chnget:k("volC"), ar*chnget:k("volC")
chnclear "reverbsendl", "reverbsendr"
	endin
	
	instr ReverbG
arevl chnget "reverbsendl"
arevr chnget "reverbsendr"
kfb init 0.98
kcf init 8000
al, ar reverbsc arevl, arevr, kfb, kcf
outs al*chnget:k("volG"), ar*chnget:k("volG")
chnclear "reverbsendl", "reverbsendr"
	endin

</CsInstruments>
<CsScore>
f0 z
i "Trigger"      0 [60*60*24*7] 
i "Stop"         0 [60*60*24*7]
i "DelayVerb1"   0 [60*60*24*7] 
i "DelayVerb2"   0 [60*60*24*7] 
i "ReverbC"      0 [60*60*24*7] 
i "ReverbG"      0 [60*60*24*7]   
</CsScore>
</CsoundSynthesizer>