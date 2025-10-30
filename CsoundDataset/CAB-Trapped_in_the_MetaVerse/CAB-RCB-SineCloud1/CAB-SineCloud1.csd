<Cabbage> bounds(0, 0, 0, 0)
form caption("SineCloud") size(333, 222), guiMode("queue") pluginId("rb33")

button  bounds(24, 150, 71, 40)   channel("trig") text("Trigger") textColour("white")
button  bounds(236, 150, 71, 40)   channel("stop") text("Stop") textColour("white")

hslider bounds(15, 52, 156, 40) channel("dur") range(1, 180, 10) text("Duration") textColour("white")
hslider bounds(176, 52, 150, 40) channel("rate") range(.1, 333, 60) text("Rate") textColour("white")

hslider bounds(176, 10, 149, 35) channel("trns") range(0.1, 6, 1)  text("Transpose") textColour("white")
hslider bounds(16, 10, 154, 37) channel("vol") range(0.0, 1, 0.75) text("Volume") textColour("white")
hslider bounds(174, 98, 151, 38) channel("del") range(0.0, 1, 0.6)   text("Delay") textColour("white")
hslider bounds(14, 98, 156, 37) channel("rev") range(0.0, 1, .7)   text("Reverb") textColour("white")

combobox bounds(100, 174, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") text("MidBellVerb", "MidBellDry", "HiBellVerb", "HiBellDry", "LoDry", "LoWet", "SlowMidVerb", "FastMidVerb")
filebutton bounds(102, 144, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(166, 144, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
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

; Inspired by and Modeled on Joaquim Heintz's SineCloud

chn_a "reverbsend", 3
chn_a "delaysend", 3

seed 0

    instr Trigger
kTrig chnget "trig"
  if changed(kTrig) == 1 then
    event "i", "Schedule", 0, chnget:i("dur")
  endif
kStop chnget "stop"
  if changed(kStop) == 1 then
    turnoff2 "Schedule", 0, 0
    turnoff2 "Cloud", 0, 1
    turnoff2 "PercSine", 0, 2
  endif
    endin

    instr Schedule
kTick init 1
kTick  metro chnget:k("rate")
kSchedRnd trandom kTick, 1, 5
  if (int(kSchedRnd) == 1) then
	    kSched = 1
    else kSched = 0
  endif
gkDurRnd trandom kTick, .1, .5
schedkwhen kSched, 0, 1, "Cloud", 0, gkDurRnd
    endin

instr Cloud
 iMinDens random 0.1,1
 iMaxDens random 1,5
 iMinNotes[] fillarray 60, 62, 64, 65, 67, 60, 69, 71 
 iMinIndex random 0, 8
 iMinIndex = int(iMinIndex)
 iMinPitch = iMinNotes[iMinIndex]
 iMidNotes[] fillarray 60, 69, 70, 68, 65, 66, 64, 63 
 iMidIndex random 0, 8
 iMidIndex = int(iMidIndex)
 iMidPitch = iMidNotes[iMidIndex]
 iMaxNotes[] fillarray 60, 72, 74, 76, 77, 80, 82, 84
 iMaxIndex random 0, 8
 iMaxIndex = int(iMaxIndex)
 iMaxPitch = iMaxNotes[iMaxIndex]
 iMinVolDb random 40, 60
 iMaxVolDb random 20, 10
 iMinDur   random 0.1, 0.5
 iMaxDur   random 0.5, 4
 
 iFirstDens = (iMinDens+iMaxDens) / 2
 kDensity init iFirstDens

 kNextEvent metro kDensity
 if kNextEvent == 1 then
  kVolume random -1*(iMinVolDb), -1*(iMaxVolDb)
  kDuration random iMinDur, iMaxDur
  schedulek "PercSine", 0, kDuration, kVolume, iMinPitch
  schedulek "PercSine", 0, kDuration, kVolume, iMidPitch
  schedulek "PercSine", 0, kDuration, kVolume, iMaxPitch
  kDensity random iMinDens, iMaxDens
 endif
endin

    instr PercSine
iDb = p4 
iMidiPitch = p5 
iAttackTime random 1/1000, 3/1000 
aEnv transeg 0, iAttackTime, 4, ampdb(iDb), p3-iAttackTime, -4, 0
aEnvRndDb randi 2, 20, 2
aSine poscil aEnv*ampdb(aEnvRndDb), cpsmidinn(iMidiPitch)*chnget:i("trns")
aSine linen aSine, .003, p3, 0
kpan rspline -1, 1, 0.5, 4
aL,aR pan2 aSine, kpan
outs aL*chnget:k("vol"), aR*chnget:k("vol")
chnmix aL*chnget:k("del"), "delaysend"
chnmix aR*chnget:k("rev"), "reverbsend"
    endin

	instr Delay
adel chnget "delaysend"
khpf init 300
klpf init 8000
klpfmod1 lfo 1000, 4
klpfmod2 lfo 1000, 5
kfb randi .5, 2
irates[] fillarray 1, 2, 4, 8, 16, 32, 64, 128, 256
kdelayrate random 0, 8
afb init .7
adel butterhp adel, khpf
kdelay = ((gkDurRnd*irates[kdelayrate]))*4*100
adelayl vdelay adel+afb, kdelay, 500
adelayr vdelay adel+afb, kdelay*0.75, 500
adelayl butterlp adelayl, klpf+klpfmod1
adelayr butterlp adelayr, klpf+klpfmod2
afb = ((adelayl + adelayr)*0.5) * kfb
aoutl, aoutr freeverb adelayl, adelayr, 0.94, 0.4, sr, 0
al ntrpol adelayl, aoutl, 0.5
ar ntrpol adelayr, aoutr, 0.5
outs al*0.36*chnget:k("vol"), ar*0.45*chnget:k("vol")
chnclear "delaysend"
	endin

	instr Reverb
arev chnget "reverbsend"
kfb init 0.98
kcf init 12000
arevl butterhp arev, 300
al, ar reverbsc arev, arev, kfb, kcf, sr, 0.5, 1
outs al*chnget:k("vol"), ar*chnget:k("vol")
chnclear "reverbsend"
	endin

</CsInstruments>
<CsScore>
f0 z
i "Trigger" 0 [60*60*24*7] 
i "Delay"   0 [60*60*24*7] 
i "Reverb"  0 [60*60*24*7]  
</CsScore>
</CsoundSynthesizer>