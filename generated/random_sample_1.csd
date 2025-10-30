<CsoundSynthesizer>
<CsOptions>
-o random_sample_1.wav -W
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

; -----------------------------
; Generated global parameters
; -----------------------------
gk_p0 init 1.000000
gk_p1 init 0.644120
gk_p2 init 0.280038
gk_p3 init -1.000000
gk_p4 init -1.000000
gk_p5 init 0.000000
gk_p6 init 0.000000
gk_p7 init 0.000000

; -----------------------------
; Tables
; -----------------------------
giSine ftgen 1, 0, 4096, 10, 1
giTri  ftgen 2, 0, 4096, 10, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1

; -----------------------------
; Synth voice
; -----------------------------
instr 1
  kBaseHz   = 220 + gk_p0 * 220
  kDetune   = 0.5 + (gk_p1 * 0.5)
  kCutoff   = 800 + abs(gk_p2) * 5000
  kRes      = 0.1 + abs(gk_p6) * 0.85
  kNoiseAmt = gk_p5
  if (kNoiseAmt < 0) then
    kNoiseAmt = 0
  endif
  kNoiseAmt = kNoiseAmt * 0.4
  kChDepth  = gk_p7
  if (kChDepth < 0) then
    kChDepth = 0
  endif
  kChDepth = kChDepth * 0.008
  kWidth    = 0.2 + (abs(gk_p4) * 0.8)

  iAtk = 0.005 + abs(i(gk_p3)) * 0.150
  iDec = 0.050 + abs(i(gk_p3)) * 0.300
  iSus = 0.4  + abs(i(gk_p3)) * 0.5
  iRel = 0.200 + abs(i(gk_p3)) * 0.500
  kEnv linsegr 0, iAtk, 1, iDec, iSus, iRel, 0

  aOsc1 vco2 0.35, kBaseHz * (1 + kDetune*0.01)
  aOsc2 vco2 0.35, kBaseHz * (1 - kDetune*0.01)
  aSub  vco2 0.25, kBaseHz * 0.5

  aNoise rand 1
  aNoise = aNoise * kNoiseAmt

  aMix = aOsc1 + aOsc2 + aSub + aNoise

  aFilt moogladder aMix, kCutoff, kRes

  ; Simple stereo output with basic chorus effect
  aL = aFilt * kEnv * 0.7
  aR = aFilt * kEnv * 0.7
  
  ; Add simple stereo widening
  aOutL = aL * (1 + kChDepth)
  aOutR = aR * (1 - kChDepth)

  outs aOutL, aOutR
endin

</CsInstruments>
<CsScore>
i1 0 5
e
</CsScore>

</CsoundSynthesizer>
<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>100</x>
 <y>100</y>
 <width>320</width>
 <height>240</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="background">
  <r>240</r>
  <g>240</g>
  <b>240</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
