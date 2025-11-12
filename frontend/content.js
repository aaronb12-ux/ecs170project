document.getElementById('extract-button').addEventListener('click', () => {
    browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
        const activeTab = tabs[0];
        
        browser.tabs.executeScript(activeTab.id, {
			    code: `
					(function() {
						let jobDescriptionDiv = document.getElementById('jobDescriptionText');
						return jobDescriptionDiv ? jobDescriptionDiv.innerText : "Job description not found.";
					})();
				`
        }).then((results) => {
            document.getElementById('result').innerText = results[0] || "No result returned.";
        }).catch(error => {
            console.error(error);
            document.getElementById('result').innerText = error;
        });
    });
});

