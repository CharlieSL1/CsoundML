<Cabbage> bounds(0, 0, 0, 0)
form caption("Crackle1") size(453, 180), guiMode("queue") pluginId("rb36")
button  bounds(56, 104, 71, 40)   channel("trig") text("Trigger") textColour("white")
button  bounds(302, 108, 71, 40)   channel("stop") text("Stop") textColour("white")
hslider bounds(288, 42, 148, 37) channel("filt") range(.15, 5, 1, .5, .01) text("Filter") textColour("white")
hslider bounds(290, 6, 148, 32) channel("dur") range(.001, 3000, 300, .5, .01) text("Duration") textColour("white")
hslider bounds(144, 6, 147, 33) channel("trans") range(0.1, 8, 1, .5, .01)  text("Transpose") textColour("white")
hslider bounds(0, 6, 146, 32) channel("vol") range(0.0, 1, 0.4, .5, .01) text("Volume") textColour("white")
hslider bounds(0, 46, 147, 33) channel("rev") range(0.0, 1, .7, .5, .01)   text("Reverb") textColour("white")
hslider bounds(141, 44, 149, 35) channel("del") range(0.0, 1, .7, .5, .01)   text("Delay") textColour("white")
combobox bounds(148, 128, 125, 29), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1") text("BellVerb")
filebutton bounds(150, 98, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(214, 98, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")
</Cabbage>
<CsoundSynthesizer>
<CsOptions>
-dm0
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 32
nchnls = 2
0dbfs  = 1

gaDelL  init 0
gaDelR  init 0
gaRvbL  init 0
gaRvbR  init 0

    instr Trigger
kTrig chnget "trig"
  if changed(kTrig) == 1 then
    event "i", "Dust", 0.0, chnget:i("dur")
  endif
kStop chnget "stop"
  if changed(kStop) == 1 then
    event "i", "TurnOff", 0.0, .1
  endif
      endin

    instr TurnOff
turnoff2 "Dust", 0, 1
    endin
  
    instr Dust
iGain init .34
kdens rspline .3, 11, .1, 4
asig  dust2 iGain, kdens*chnget:k("trans")
kfco rspline 200, 5000, .1, 2
kres rspline 0, .58, .2, 12
kdist rspline 0, 2, .2, 2
asig lpf18 asig, (100+kfco)*chnget:k("filt"), 0+kres, 0+kdist
kpan rspline -1, 1, .1, 6
kgainL rspline 0, 2.3, .001, .12  
kgainR rspline 0, 2.7, .002, .13 
aL =  asig * kpan * kgainL
aR =  asig * (1-kpan) * kgainR
aL = aL * chnget:k("vol")
aR = aR * chnget:k("vol")
outs aL, aR
kDelL rspline .1, .35, 0.3, 1
kDelR rspline .1, .34, 0.4, 2
gaDelL  += aL * chnget:k("del")
gaDelR  += aR * chnget:k("del")
kVerbL rspline .1, .43, 0.1, 3
kVerbR rspline .1, .45, 0.2, 4
gaRvbL  += aL * chnget:k("rev")
gaRvbR  += aR * chnget:k("rev")
    endin

    instr Delay  ; Based on Scott Daughtry's Ghost Bell Delay
iTimeL = .6
iTimeR = .7
denorm(gaDelL,gaDelR)
aDelL multitap gaDelL, iTimeL, .95, iTimeL*3, .70, iTimeL*5, .43, iTimeL*7, .15
aDelR multitap gaDelR, iTimeR, .83, iTimeR*4, .57, iTimeR*6, .31, iTimeR*8, .05
outs aDelL, aDelR
clear gaDelL, gaDelR
    endin

    instr Reverb
denorm(gaRvbL,gaRvbR)
aL,aR   reverbsc gaRvbL, gaRvbR, .98, 12000
outs aL, aR
clear gaRvbL, gaRvbR
    endin
</CsInstruments>
<CsScore>
f0 z
i"Trigger" 0 [60*60*24*7] 
i"Delay"   0 [60*60*24*7] 
i"Reverb"  0 [60*60*24*7] 
</CsScore>
</CsoundSynthesizer>