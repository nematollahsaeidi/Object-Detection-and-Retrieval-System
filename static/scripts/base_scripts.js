function myFunction(id,check_dominant) {
	var form = document.createElement('form');
	form.method = "post";
	form.action = "http://172.16.107.251:5001/";
//	############
	var hiddenField_check_dominant = document.createElement('input');
	hiddenField_check_dominant.type = 'hidden';
	hiddenField_check_dominant.name = 'check_dominant';
	hiddenField_check_dominant.value = check_dominant;
	hiddenField_check_dominant.id = 'check_dominant';
//	############
	var hiddenField = document.createElement('input');
	hiddenField.type = 'hidden';
	hiddenField.name = 'id';
	hiddenField.value = id;
	hiddenField.id = 'id';
	form.appendChild(hiddenField);
	form.appendChild(hiddenField_check_dominant);
	document.body.appendChild(form);
	form.submit();
}

// Script to open and close sidebar
function w3_open() {
  document.getElementById("mySidebar").style.display = "block";
  document.getElementById("myOverlay").style.display = "block";
}

function w3_close() {
  document.getElementById("mySidebar").style.display = "none";
  document.getElementById("myOverlay").style.display = "none";
}

// Modal Image Gallery
function onClick(element) {
  document.getElementById("img01").src = element.src;
  document.getElementById("modal01").style.display = "block";
  var captionText = document.getElementById("caption");
  captionText.innerHTML = element.alt;
}