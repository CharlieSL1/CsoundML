<Cabbage> bounds(0, 0, 0, 0)
form caption("Random Sines") size(800, 400), guiMode("queue") pluginId("rsin")

button  bounds(6, 12, 80, 42) channel("trigger") text("Trigger") textColour("white")
checkbox bounds(196, 18, 27, 25), channel("reTrigger"), , fontColour:0("white")
label    bounds(96, 20, 95, 19), text("ReTrigger"), fontColour("white") 
hslider bounds(236, 6, 175, 50) channel("rate") range(1, 10, 4, 1, 0.001) text("ReTrigRate") textColour("white")

hslider bounds(452, 6, 150, 50) channel("maxDur") range(.0001, 1, .5, 1, 0.001) text("Max Dur") textColour("white")
hslider bounds(612, 6, 150, 50) channel("numInstrs") range(1, 30, 20, 1, 0.001) text("Num Voices") textColour("white")

hslider bounds(70, 132, 150, 50) channel("1frqLow") range(40, 1000, 100, 1, 0.001) text("1FreqLow") textColour("white")
hslider bounds(230, 132, 150, 50) channel("1frqHi") range(30, 2000, 500, 1, 0.001) text("1FreqHi") textColour("white")
hslider bounds(70, 190, 150, 50) channel("2frqLow") range(40, 1000, 200, 1, 0.001) text("2FreqLow") textColour("white")
hslider bounds(230, 190, 150, 50) channel("2frqHi") range(50, 2000, 600, 1, 0.001) text("2FreqHi") textColour("white")

hslider bounds(402, 132, 150, 50) channel("1durLow") range(.001, 1, .3, 1, 0.001) text("1DurLow") textColour("white")
hslider bounds(568, 132, 150, 50) channel("1durHi") range(.002, 1, .4, 1, 0.001) text("1DurHi") textColour("white")
hslider bounds(404, 190, 150, 50) channel("2durLow") range(.002, 1, .4, 1, 0.001) text("2DurLow") textColour("white")
hslider bounds(572, 190, 150, 50) channel("2durHi") range(.003, 1, .5, 1, 0.001) text("2DurHi") textColour("white")

hslider bounds(166, 68, 150, 50) channel("transpose") range(0.5, 4, 1, 1, 0.001) text("Transpose") textColour("white")
hslider bounds(328, 68, 150, 50) channel("pan") range(0, 1, 0.5, 1, 0.001) text("FX Pan") textColour("white")
hslider bounds(10, 68, 150, 50) channel("volume") range(0, .7, .5, 1, 0.001) text("Volume") textColour("white")
hslider bounds(488, 66, 150, 51) channel("chorus") range(0, 1, .6, 1, 0.001) text("Chorus") textColour("white")
hslider bounds(646, 68, 150, 50) channel("reverb") range(0, 1, .7, 1, 0.001) text("Reverb") textColour("white")

combobox bounds(348, 366, 100, 25), populate("*.snaps"), channelType("string") automatable(0) channel("combo31")  value("1")
filebutton bounds(282, 368, 60, 25), text("Save", "Save"), populate("*.snaps", "test"), mode("named preset") channel("filebutton32")
filebutton bounds(454, 364, 60, 25), text("Remove", "Remove"), populate("*.snaps", "test"), mode("remove preset") channel("filebutton33")

</Cabbage>

<CsoundSynthesizer>
<CsOptions>
-n -dm0
</CsOptions>
<CsInstruments>
; Initialize the global variables. 
ksmps = 32
nchnls = 2
0dbfs = 1

seed 0

gaChorL init 0
gaChorR init 0

gaRevL init 0
gaRevR init 0

gisine ftgen 0, 0, 2^10, 10, 1

instr 1

    kTrig   chnget "trigger"
    kReTrig chnget "reTrigger"
              
    if kReTrig == 1 then
        kTrig metro chnget:k("rate")
    endif

    if changed(kTrig) == 1 then
        event "i", "Generate", 0, chnget:i("maxDur")
    endif
   
endin


instr Generate

iNumInstrs = chnget:i("numInstrs")

indx = 0

gi1frqLow = chnget:i("1frqLow")+rnd(10)
gi1frqHi = chnget:i("1frqHi")+rnd(40)
gi1durLow = chnget:i("1durLow")+rnd(.12)
gi1durHi = chnget:i("1durHi")+rnd(.16)

gi2frqLow = chnget:i("2frqLow")+rnd(4)
gi2frqHi = chnget:i("2frqHi")+rnd(44)
gi2durLow = chnget:i("2durLow")+rnd(.12)
gi2durHi = chnget:i("2durHi")+rnd(.18)
gi2strtHi = chnget:i("maxDur")-gi2durHi

        loop:

        iamp  = 1/iNumInstrs

        istrt1 = 0
        idur1  random gi1durLow,gi1durHi
        ifreq1 random gi1frqLow,gi1frqHi
        ipan1  random 0,1

        istrt2 random 2,gi2strtHi
        idur2  random gi2durLow,gi2durHi
        ifreq2 random gi2frqLow,gi2frqHi
        ipan2  random 0,1

        event_i "i", "PlaySines", istrt1, idur1, iamp, ifreq1, ipan1
        event_i "i", "PlaySines", istrt2, idur2, iamp, ifreq2, ipan2
 
        loop_lt indx, 1, iNumInstrs, loop
        
endin

instr PlaySines

gkVol    = chnget:k("volume")

iRevL   = .54+rnd(.3)
iRevR   = .38+rnd(.28)
iChorL  = .6+rnd(.3)
iChorR  = .56+rnd(.33)
ishape1 = int(rnd(12))
ishape2 = -1*(int(rnd(12)))
ipeak       random  0, 1 

asigL       poscil3 p4, p5*chnget:i("transpose"), gisine
asigR       poscil3 p4, p5*chnget:i("transpose"), gisine
aenv        transeg 0, p3*ipeak, ishape1, 1, p3-p3*ipeak, ishape2, 0
aDry        =  (asigL + asigR)*aenv
aDryL,aDryR pan2    aDry, p6;*chnget:k("pan")
aWetL,aWetR pan2    aDry, chnget:k("pan")
            outs    aDryL*gkVol, aDryR*gkVol
        
        vincr   gaRevL,  aWetL*iRevL
        vincr   gaRevR,  aWetR*iRevR  
        vincr   gaChorL, aWetL*iChorL
        vincr   gaChorR, aWetR*iChorR  
endin

instr   Chorus  ; from Lazzarini

gkChor    = chnget:k("chorus")
            denorm gaChorL, gaChorR
 amod1  =   randi:a(3,.75)+oscili(2,.35)+31
 amod2  =   randi:a(2,.65)+oscili(3,.55)+29
 aL     =   vdelay(gaChorL,amod1,35)
 aR     =   vdelay(gaChorR,amod2,35)
            outs  aL*chnget:k("volume")*chnget:k("chorus"), aR*chnget:k("volume")*chnget:k("chorus")
            clear gaChorL, gaChorR
endin

instr   Reverb
gkChor    = chnget:k("reverb")
        denorm gaRevL, gaRevR
aL, aR  reverbsc gaRevL, gaRevR, 0.85, 12000, sr, 0.5, 1
            outs  aL*chnget:k("volume")*chnget:k("reverb"), aR*chnget:k("volume")*chnget:k("reverb")
        clear gaRevL, gaRevR
endin   

</CsInstruments>

<CsScore>
;causes Csound to run for about 7000 years...
f0 z
;starts instrument 1 and runs it for a week
i1 0 [60*60*24*7] 
i "Chorus" 0 [60*60*24*7]
i "Reverb" 0 [60*60*24*7]  
</CsScore>

</CsoundSynthesizer>