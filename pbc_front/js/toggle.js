//toggle.js
function toggleText(event)
{
    var text = event.textContent || event.innerText;
    if(text == '등록') {
        event.innerHTML = '제목';
    }
    else {
        event.innerHTML = '등록';
    }
}