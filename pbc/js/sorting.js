const searchBox = document.querySelector(".search-box");
const searchBtn = document.querySelector(".search-icon");
const cancelBtn = document.querySelector(".cancel-icon");
const searchInput = document.querySelector("input");
// const searchData = document.querySelector(".search-data");
// const settingBtn = document.querySelector(".setting");
// const userBtn = document.querySelector(".user");
const sortBtn = document.querySelector(".sortbtn");
searchBtn.onclick =()=>{
    searchBox.classList.add("active");
    searchBtn.classList.add("active");
    searchInput.classList.add("active");
    cancelBtn.classList.add("active");
    // settingBtn.classList.add("active");
    // userBtn.classList.add("active");
    sortBtn.classList.add("active");
    searchInput.focus();
    if(searchInput.value != ""){
        var values = searchInput.value;
        // searchData.classList.remove("active");
        // searchData.innerHTML = "You just typed " + "<span style='font-weight: 500;'>" + values + "</span>";
    }
    // else{
    //     searchData.textContent = "";
    // }
}
cancelBtn.onclick =()=>{
    searchBox.classList.remove("active");
    searchBtn.classList.remove("active");
    searchInput.classList.remove("active");
    cancelBtn.classList.remove("active");
    // searchData.classList.toggle("active");
    // settingBtn.classList.remove("active");
    // userBtn.classList.remove("active");
    sortBtn.classList.remove("active");
    searchInput.value = "";
}