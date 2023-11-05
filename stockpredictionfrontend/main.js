const apiUrl = "http://127.0.0.1:8000/";

function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const CSRFTOKEN = getCookie("csrftoken");
const headers = {
  "X-CSRFToken": CSRFTOKEN,
};

let graphData = {};
document
  .getElementById("stock-get-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent the default form submission

    const stockTag = document.querySelector('input[name="stockTag"]').value;
    console.log(stockTag)
    // You can use the stockTag value to make your Axios GET request here
    try {
      graphData = await axios.get(apiUrl + "get-stock-data/", {
        params: { stock_code: stockTag },
        headers,
      });
      console.log(graphData)
    } catch (error) {
      console.log(error);
    }
  });

function createLineChart(data) {
    // Create an SVG element
    let loadData = (data) => {
        const dates = data[0];
        const closingPrices = data[1];
        const volumes = data[2];
    
        // Now, you can transform the data into the desired format
        const transformedData = dates.map((date, index) => ({
            date: new Date(date),
            close: closingPrices[index],
            volume: volumes[index]
        }));
    
        return transformedData;
    };

    let data = loadData(graphData)

    // Define margins and dimensions
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };
    const width = 1600 - margin.left - margin.right;
    const height = 800 - margin.top - margin.bottom;

    // Create scales
    const xScale = d3.scaleTime()
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .range([height, 0]);

    const svg = d3.select('#container')
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("heigh", height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    xScale.domain(d3.extent(data, d => d.Date))
    yScale.domain([0, d3.max(data, d => d.close)])
}

// Call the function with your data
createLineChart(graphData);
