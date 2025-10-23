const API_BASE = '' // e.g. http://localhost:8000
const apiKey = apiKeyInput.value.trim() || DEFAULT_API_KEY
try{
const form = new FormData()
form.append('file', file, file.name)
const res = await fetch((API_BASE||'') + '/predict', {
method: 'POST',
headers: {'x-api-key': apiKey},
body: form
})
if(!res.ok){
const err = await res.json().catch(()=>({detail: 'Unknown error'}))
throw new Error(err.detail || err.message || 'Prediction failed')
}
const data = await res.json()
displayResults(data)
addHistory(previewImg.src, data.class_name, data.confidence)
}catch(err){
errorDiv.textContent = err.message
}finally{ showSpinner(false) }



function displayResults(data){
resultsDiv.classList.remove('empty')
resultsDiv.innerHTML = `<strong>${data.class_name}</strong> — ${ (data.confidence*100).toFixed(2)}%`
// top3
top3Div.innerHTML = ''
const preds = Object.entries(data.predictions).sort((a,b)=>b[1]-a[1]).slice(0,3)
preds.forEach(([label,prob])=>{
const wrap = document.createElement('div'); wrap.className='pred-bar'
const lbl = document.createElement('div'); lbl.className='pred-label'; lbl.textContent = label
const bar = document.createElement('div'); bar.className='bar'
const fill = document.createElement('div'); fill.className='bar-fill'; fill.style.width = (prob*100)+'%'
const pct = document.createElement('div'); pct.style.width='70px'; pct.textContent=(prob*100).toFixed(1)+'%'
bar.appendChild(fill); wrap.appendChild(lbl); wrap.appendChild(bar); wrap.appendChild(pct); top3Div.appendChild(wrap)
})
}


function showSpinner(on){ spinner.classList.toggle('hidden', !on) }
function clearResults(){ resultsDiv.innerHTML=''; top3Div.innerHTML=''; errorDiv.textContent='' }


// sample image (small base64 32x32 placeholder) - you can replace with a hosted sample
sampleBtn.addEventListener('click', ()=>{
fetch('https://raw.githubusercontent.com/moatazgenius/sample-images/main/cat32.png')
.then(r=>r.blob())
.then(blob=>{ const f = new File([blob],'sample.png',{type:blob.type}); handleFile(f) })
.catch(()=> alert('Sample image not available'))
})


// webcam capture (advanced)
webcamBtn.addEventListener('click', async ()=>{
try{
const stream = await navigator.mediaDevices.getUserMedia({video:{width:320,height:240}})
const video = document.createElement('video'); video.autoplay=true; video.srcObject = stream
const modal = document.createElement('div'); modal.className='modal'
const capBtn = document.createElement('button'); capBtn.textContent='Capture'
modal.appendChild(video); modal.appendChild(capBtn); document.body.appendChild(modal)
capBtn.addEventListener('click', ()=>{
const canvas = document.createElement('canvas'); canvas.width=32; canvas.height=32
const ctx = canvas.getContext('2d'); ctx.drawImage(video,0,0,32,32)
canvas.toBlob(blob=>{
const f = new File([blob],'webcam.png',{type:'image/png'}); handleFile(f)
})
stream.getTracks().forEach(t=>t.stop()); document.body.removeChild(modal)
})
}catch(e){ alert('Webcam not available or permission denied') }
})


// history
function addHistory(imgSrc,label,conf){
const item = document.getElementById('history-it')}