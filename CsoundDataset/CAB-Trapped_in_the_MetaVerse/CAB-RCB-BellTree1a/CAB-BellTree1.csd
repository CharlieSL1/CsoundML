<Cabbage> bounds(0, 0, 0, 0)
form caption("SineCloud") size(433, 180), guiMode("queue") pluginId("rb33")

button  bounds(56, 104, 71, 40)   channel("trig") text("Trigger") textColour("white")
button  bounds(302, 108, 71, 40)   channel("stop") text("Stop") textColour("white")

hslider bounds(56, 52, 150, 40) channel("dur") range(.001, 3000, 300) text("Duration") textColour("white")

hslider bounds(256, 6, 150, 40) channel("trns") range(0.1, 8, 1)  text("Transpose") textColour("white")
hslider bounds(14, 10, 146, 32) channel("vol") range(0.0, 1, 0.4) text("Volume") textColour("white")
hslider bounds(216, 52, 158, 40) channel("rev") range(0.0, 1, .7)   text("Reverb") textColour("white")

combobox bounds(148, 128, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") text("BellVerb")
filebutton bounds(150, 98, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(214, 98, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
</Cabbage>
<CsoundSynthesizer>
<CsOptions>
-dm0
</CsOptions>
<CsInstruments>

; a modification and simplification of Steven Yi's 'cyclic bells'

nchnls = 2
0dbfs  = 1

chn_a "reverbsendl", 3
chn_a "reverbsendr", 3

seed 0

gi_hit_probability	= 0.20

    instr Trigger
kTrig chnget "trig"
  if changed(kTrig) == 1 then
    event "i", "Schedule", 0.0, chnget:i("dur"), 10, .1
    event "i", "Schedule", 0.3, chnget:i("dur"), 11, .1
    event "i", "Schedule", 2.5, chnget:i("dur"), 12, .1
    event "i", "Schedule", 4.8, chnget:i("dur"), 12, .1
  endif
kStop chnget "stop"
  if changed(kStop) == 1 then
    event "i", "TurnOff", 0.0, .1
  endif
      endin

    instr TurnOff
turnoff2_i "Schedule", 0, 0
turnoff2_i "Bell", 0, 1
    endin
   
    instr Schedule
iTableNum	= p4
iBaseAmp 	= p5
kcount  init 5 * kr
kdur	init 4 * kr
kHit    init 0
kIndex  init 0
if (kcount < kdur) kgoto counterup	
	kHit rnd31 .5, 0
	kHit = kHit + .5
	if (kHit > gi_hit_probability) kgoto nohit
		kpch1 table 0, iTableNum
		kpch2 table	kIndex, iTableNum		
		event "i", "Bell", 0, 10, kpch1, kpch1
		event "i", "Bell", 0, 10, kpch2, kpch2
		kdur = (kHit * 3) + 3
		kdur = kdur * kr
nohit:
		kIndex 	= kIndex + 1
		if (kIndex < 5) kgoto noZeroing
			kIndex = 0
noZeroing:
	kcount = 0
counterup:
	kcount = kcount + 1
    endin

    instr Bell
ipch1 = cpspch(p4)
ipch2 = cpspch(p5)
kpch line ipch1*chnget:i("trns"), p3, ipch2*chnget:i("trns")
kpch = kpch*rnd(.5)
iamp = .02+rnd(.05)
kc1 = 2+rnd(7)
kc2 = 1+rnd(5)
kvdepth = .3*rnd(.3)
kvrate = 3.3*rnd(3)
asig fmbell iamp, kpch, kc1, kc2, kvdepth, kvrate
kpan rspline -1, 1, 0.5, 4
aL,aR pan2 asig, kpan
aL = aL*chnget:k("vol")
aR = aR*chnget:k("vol")
outs aL, aR
chnmix aL*rnd(.20)*chnget:k("vrb"), "reverbsendl"
chnmix aR*rnd(.21)*chnget:k("vrb"), "reverbsendr"
gkfb = rnd(.32)
gkcf = rnd(6000)
    endin

	instr Reverb
arevl chnget "reverbsendl"
arevr chnget "reverbsendr"
denorm(arevl,arevr)
kfb init .67
kcf init 2000
al,ar reverbsc arevl, arevr, kfb+gkfb, kcf+gkcf
outs al, ar
chnclear "reverbsendl","reverbsendr"
	endin
</CsInstruments>
<CsScore>
f0 z
f1 0 65536 10 1
f10 0 8 -2 10.04 10.06 10.09 10.01 10.07 0 0 0
f11 0 8 -2 10.07 10.08 10.04 10.02 10.06 0 0 0
f12 0 8 -2 10.02 10.01 10.11 10.05 10.03 0 0 0

i "Trigger" 0 [60*60*24*7] 
i "Reverb"  0 [60*60*24*7] 
</CsScore>
</CsoundSynthesizer>