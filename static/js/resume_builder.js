
const baseUrl = window.location.origin;



async function generateResults() {


    const resume = document.getElementById("pdfInput").files[0];
    const position = document.getElementById("position").value.trim(); 
    const company = document.getElementById("companyName").value.trim();
    const jobDesc = document.getElementById("jobDescription").value.trim();
    const cvContent = document.getElementById("cvContent").value.trim();

    if (!resume & !cvContent) {
        alert("Please upload a PDF of your resume or fill in the CV content.");
        return;
    }

    if (!company || !jobDesc || !position) {
      alert("Please fill in all required fields.");
      return;
    }
    

    const formData = new FormData();
    formData.append("resume", resume);
    formData.append("position", position);
    formData.append("company", company);
    formData.append("jobDesc", jobDesc);
    formData.append("cvContent", cvContent);
    console.log("here")

    

    let response = await fetch(baseUrl + "/generate-results", {
        method: "POST",
        headers: { Accept: "application/json" },
        body: formData
    });
    
    data = await response.json();
    if (response.ok) {
        
    
        console.log("Cover Letter: here:", response);

        document.getElementById("coverLetter").innerHTML = data["coverLetter"];
        document.getElementById("cvImprovements").innerHTML = data["cvImprovements"];
    }
    
    else{
        alert("Error generating results. Please try again.");
    }

}