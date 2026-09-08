/* ============================================================
   AGRISHIELD AI - CHATGPT FRONTEND CONTROLLER (ES6 JAVASCRIPT)
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    // UI Layout Controls
    const themeToggle = document.getElementById("theme-toggle");
    const sidebar = document.getElementById("sidebar");
    const openSidebarBtn = document.getElementById("open-sidebar-btn");
    const closeSidebarBtn = document.getElementById("close-sidebar-btn");
    const newDiagnosisBtn = document.getElementById("new-diagnosis-btn");
    const historyScansList = document.getElementById("history-scans-list");
    const sidebarBackdrop = document.getElementById("sidebar-backdrop");
    const welcomeDropzone = document.getElementById("welcome-dropzone");
    
    // Model Selector Dropdown Controls
    const modelDropdownBtn = document.getElementById("model-dropdown-btn");
    const activeModelName = document.getElementById("active-model-name");
    const modelDropdownMenu = document.getElementById("model-dropdown-menu");
    
    // Chat Feed Elements
    const chatFeed = document.getElementById("chat-feed");
    const welcomeScreen = document.getElementById("welcome-screen");
    const messageThread = document.getElementById("message-thread");
    
    // Bottom Input Pill Bar Elements
    const fileInput = document.getElementById("file-input");
    const attachFileBtn = document.getElementById("attach-file-btn");
    const openCameraBtn = document.getElementById("open-camera-btn");
    const chatTextInput = document.getElementById("chat-text-input");
    const sendDiagnosisBtn = document.getElementById("send-diagnosis-btn");
    
    // Specimen Thumbnail Previews
    const selectedImageBar = document.getElementById("selected-image-bar");
    const inputImageThumbnail = document.getElementById("input-image-thumbnail");
    const selectedFilename = document.getElementById("selected-filename");
    const clearSelectedImage = document.getElementById("clear-selected-image");
    const thumbnailLaserScanner = document.getElementById("active-laser-line");
    
    // Web Camera Stream Overlays
    const cameraOverlay = document.getElementById("camera-overlay");
    const cameraStream = document.getElementById("camera-stream");
    const cameraCanvas = document.getElementById("camera-canvas");
    const captureSnapshotBtn = document.getElementById("capture-snapshot-btn");
    const closeCameraBtn = document.getElementById("close-camera-btn");
    
    const systemStatusIndicator = document.getElementById("system-status-indicator");

    // Application States
    let selectedModelId = "InceptionV3_Cotton_Mamba";
    let activeImageFile = null;
    let cameraStreamTrack = null;
    let diagnosisHistoryList = [];
    let activeModelList = [];

    // Comprehensive Botanical Diagnostic & Preventive Guide Database
    const BOTANICAL_DATABASE = {
        // --- 🌿 COTTON CROPS & FIELD WEEDS (13 CLASSES) ---
        "cotton": {
            commonName: "Cotton Crop (Healthy)",
            scientificName: "Gossypium hirsutum L.",
            risk: "safe",
            description: "Healthy cotton foliage detected. Distinct palmate 3–5 lobed leaves with vibrant green chlorophyll saturation, intact mesophyll cellular structure, and robust photosynthetic activity.",
            prevention: "Maintain scientific N-P-K fertigation schedules, avoid over-irrigation, and implement regular integrated pest management (IPM) scouting against bollworms and sucking pests."
        },
        "Amaranthus viridis": {
            commonName: "Slender Amaranth (Chulai)",
            scientificName: "Amaranthus viridis L.",
            risk: "high",
            description: "Erect annual broadleaf weed with ovate leaves and greenish flower clusters. Rapid vegetative growth aggressively competes with cotton seedlings for sunlight, nitrogen, and soil moisture.",
            prevention: "Apply pre-emergence Pendimethalin 38.7% CS within 48 hours of sowing. Early post-emergence spot application of Pyrithiobac-sodium 10% EC for selective broadleaf suppression."
        },
        "Carpetweeds": {
            commonName: "Carpetweed",
            scientificName: "Mollugo verticillata L.",
            risk: "medium",
            description: "Prostrate annual weed forming dense, circular mats across the topsoil, choking feeder roots of young cotton plants and hindering furrow irrigation flow.",
            prevention: "Early inter-row shallow hoeing or organic mulching. Apply pre-emergence Oxyfluorfen or Pendimethalin to prevent seed germination in warm, moist seedbeds."
        },
        "Cleome gynandra": {
            commonName: "Spiderwisp (Shwet Hulhul)",
            scientificName: "Cleome gynandra L.",
            risk: "medium",
            description: "Annual herbaceous weed with 5-foliolate palmate compound leaves and sticky glandular hairs. Highly drought-tolerant and acts as an alternative host for insect vectors.",
            prevention: "Early manual hoeing at 15–20 days after sowing. Directed post-emergence application of selective broadleaf graminicides where infesting young cotton rows."
        },
        "Commelina benghalensis": {
            commonName: "Bengal Dayflower (Kankawa)",
            scientificName: "Commelina benghalensis L.",
            risk: "high",
            description: "Creeping perennial weed producing both aerial blue flowers and subterranean seeds, making field eradication extremely difficult once established in cotton fields.",
            prevention: "Apply pre-emergence Flumioxazin or Pendimethalin. Directed inter-row spraying of Glyphosate using protective spray hoods to prevent herbicide drift onto cotton foliage."
        },
        "Cynodon dactylon": {
            commonName: "Bermuda Grass (Doob Ghas)",
            scientificName: "Cynodon dactylon (L.) Pers.",
            risk: "critical",
            description: "Aggressive rhizomatous and stoloniferous perennial turf grass forming impenetrable underground root nets, severely robbing cotton plants of moisture and causing stunted bolls.",
            prevention: "Deep summer mouldboard ploughing to desiccate rhizomes under hot solar rays. Apply post-emergence Quizalofop-ethyl 5% EC or Propaquizafop 10% EC at active weed growth."
        },
        "Echinochloa colona": {
            commonName: "Jungle Rice (Sawa Grass)",
            scientificName: "Echinochloa colona (L.) Link",
            risk: "critical",
            description: "Fast-tillering annual grassy weed that rapidly overtakes young cotton stands, causing severe nutrient depletion and up to 60% yield reduction if unmanaged.",
            prevention: "Apply pre-emergence Pendimethalin 30% EC. Follow with early post-emergence application of Quizalofop-p-ethyl or Fenoxaprop-p-ethyl at the 2–3 leaf weed stage."
        },
        "Morningglory": {
            commonName: "Morning Glory (Tall / Ivyleaf)",
            scientificName: "Ipomoea purpurea / hederacea",
            risk: "critical",
            description: "Aggressive climbing vine weed that coils tightly around cotton stalks, causing severe lodging, entangling mechanical pickers, and causing substantial boll rot.",
            prevention: "Pre-emergence Prometryn or Diuron application. Apply early post-emergence directed Glufosinate-ammonium with shielded nozzles before vines begin twining cotton stems."
        },
        "Nutsedge": {
            commonName: "Purple Nutsedge (Motha)",
            scientificName: "Cyperus rotundus L.",
            risk: "critical",
            description: "Perennial sedge with underground tubers and basal bulbs. Exudes allelopathic root chemicals that directly inhibit cotton taproot elongation and boll development.",
            prevention: "Pre-plant summer tillage and soil solarization. Selective post-emergence directed application of Halosulfuron-methyl 75% WDG directed strictly between cotton rows."
        },
        "PalmerAmaranth": {
            commonName: "Palmer Amaranth (Pigweed)",
            scientificName: "Amaranthus palmeri S. Watson",
            risk: "critical",
            description: "Notorious, aggressive broadleaf weed growing up to 2-3 inches per day, known for multi-herbicide resistance. Capable of devastating cotton yields by over 70%.",
            prevention: "Zero-tolerance management: destroy escapes before seed set. Use overlapping residual herbicides (Pendimethalin followed by S-metolachlor) and timely post Glufosinate sprays."
        },
        "Phyllanthus urinaria": {
            commonName: "Chamberbitter (Hazarmani)",
            scientificName: "Phyllanthus urinaria L.",
            risk: "medium",
            description: "Warm-season annual herb with miniature sensitive leaves and abundant seed capsules under branches, leading to dense re-infestation under moist cotton canopies.",
            prevention: "Apply early pre-emergence residual herbicides like Oxyfluorfen. Regular hand rogueing of isolated plants before explosive seed capsule maturation."
        },
        "Purslane": {
            commonName: "Common Purslane (Kulfa)",
            scientificName: "Portulaca oleracea L.",
            risk: "high",
            description: "Succulent prostrate broadleaf weed with fleshy reddish-green stems that retains high water content, surviving severe drought and re-rooting from cultivation fragments.",
            prevention: "Avoid fragmenting succulent stems with rototillers. Apply pre-emergence Metolachlor or Pendimethalin early. Directed spray of 2,4-D amine with strict drift shields."
        },
        "Trianthema portulacastrum": {
            commonName: "Horse Purslane (Santhi / Bishkapra)",
            scientificName: "Trianthema portulacastrum L.",
            risk: "critical",
            description: "Dominant invasive succulent broadleaf weed in cotton tracts. Germinates rapidly to create an opaque ground carpet that smothers cotton seedlings during the first 45–60 days.",
            prevention: "Strict pre-emergence application of Pendimethalin 30% EC @ 1.0 kg a.i./ha within 48h of sowing. Early post-emergence directed Pyrithiobac-sodium @ 62.5 g a.i./ha at 20–25 DAS."
        },

        // --- 🌽 CORN CROPS & FIELD WEEDS (5 CLASSES) ---
        "bluegrass": {
            commonName: "Annual Bluegrass",
            scientificName: "Poa annua L.",
            risk: "medium",
            description: "Annual Bluegrass is a competitive grassy weed that steals early-season nitrogen from corn plants.",
            prevention: "Apply pre-emergent selective herbicides in the fall. Implement dense nitrogen cover crops and address soil compaction."
        },
        "chenopodium album": {
            commonName: "White Goosefoot (Bathua)",
            scientificName: "Chenopodium album L.",
            risk: "high",
            description: "Aggressive broadleaf competitor that reduces maize crop spacing, sunlight penetration, and final yields.",
            prevention: "Apply pre-emergent Atrazine or early post-emergent Dicamba. Perform mechanical weeding during early corn growth."
        },
        "cirsium setosum": {
            commonName: "Field Thistle",
            scientificName: "Cirsium setosum (Willd.)",
            risk: "critical",
            description: "Creeping perennial weed forming dense spikes, depleting large amounts of water and blocking harvesting blades.",
            prevention: "Avoid root fragmentation during tillage. Apply systemic Clopyralid or Glyphosate during rosette crop stages."
        },
        "corn": {
            commonName: "Zea mays L. (Maize)",
            scientificName: "Zea mays L. (Healthy)",
            risk: "safe",
            description: "Healthy crop leaf spotted! High chlorophyll density, normal chloroplast counts, and no active symptoms of disease or weeds.",
            prevention: "Follow scientific NPK schedules, perform regular scouting, and optimize row width to suppress secondary weed canopies."
        },
        "sedge": {
            commonName: "Nut Sedge",
            scientificName: "Cyperus rotundus L.",
            risk: "high",
            description: "Highly persistent perennial weed with aggressive root tubers. Signals high soil water-logging or drainage blocks.",
            prevention: "Apply post-emergent Halosulfuron-methyl. Install effective field subsurface drainages and perform deep autumn tilling."
        }
    };

    // ============================================================
    // THEME CONFIGURATION
    // ============================================================
    themeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme");
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", nextTheme);
        
        const icon = themeToggle.querySelector("i");
        icon.className = nextTheme === "light" ? "fa-solid fa-moon" : "fa-solid fa-sun";
    });

    // ============================================================
    // COLLAPSIBLE SIDEBAR
    // ============================================================
    function openSidebar() {
        sidebar.classList.remove("collapsed");
        sidebar.classList.add("active");
        if (sidebarBackdrop) sidebarBackdrop.classList.add("active");
    }

    function closeSidebar() {
        sidebar.classList.remove("active");
        sidebar.classList.add("collapsed");
        if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
    }

    closeSidebarBtn.addEventListener("click", closeSidebar);
    openSidebarBtn.addEventListener("click", openSidebar);

    if (sidebarBackdrop) {
        sidebarBackdrop.addEventListener("click", closeSidebar);
    }

    // Close sidebar overlay when clicking outside on mobile
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 768) {
            if (!sidebar.contains(e.target) && !openSidebarBtn.contains(e.target) && (!sidebarBackdrop || !sidebarBackdrop.contains(e.target))) {
                closeSidebar();
            }
        }
    });

    // ============================================================
    // MODEL DROPDOWN REGISTRY SWITCHER
    // ============================================================
    modelDropdownBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        modelDropdownMenu.classList.toggle("active");
    });

    document.addEventListener("click", () => {
        modelDropdownMenu.classList.remove("active");
    });

    function setActiveModel(modelId) {
        selectedModelId = modelId;
        const model = activeModelList.find(m => m.id === modelId);
        if (model) {
            activeModelName.textContent = model.champion ? `${model.name} ★` : model.name;
            const badgeText = document.getElementById("input-crop-badge-text");
            if (badgeText) {
                badgeText.textContent = model.crop === "Cotton" ? "Cotton Mamba" : "Corn CBAM";
            }
        }
        document.querySelectorAll(".model-menu-item").forEach(item => {
            if (item.dataset.id === modelId) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });
    }

    const inputCropBadge = document.getElementById("input-crop-badge");
    if (inputCropBadge) {
        inputCropBadge.addEventListener("click", (e) => {
            e.stopPropagation();
            modelDropdownMenu.classList.toggle("active");
        });
    }

    async function loadModelsRegistry() {
        try {
            const response = await fetch("/api/models");
            if (!response.ok) throw new Error("Registry server down");
            activeModelList = await response.json();
            
            // Set active model trigger label
            const defaultModel = activeModelList.find(m => m.id === selectedModelId) || activeModelList[0];
            if (defaultModel) {
                setActiveModel(defaultModel.id);
            }

            modelDropdownMenu.innerHTML = "";

            const cottonModels = activeModelList.filter(m => m.crop === "Cotton");
            const cornModels = activeModelList.filter(m => m.crop === "Corn");

            function renderModelItem(model) {
                const isActive = model.id === selectedModelId ? "active" : "";
                const menuItem = document.createElement("div");
                menuItem.className = `model-menu-item ${isActive}`;
                menuItem.dataset.id = model.id;
                
                const champHtml = model.champion ? `<span class="model-champ-badge">Champion</span>` : "";

                menuItem.innerHTML = `
                    <div class="model-menu-item-header">
                        <span class="model-menu-item-title">${model.name}</span>
                        ${champHtml}
                    </div>
                    <div class="model-menu-item-desc">Params: ${model.params} | Acc: ${model.accuracy} | Epochs: ${model.epochs}</div>
                `;
                
                menuItem.addEventListener("click", () => {
                    setActiveModel(model.id);
                });
                
                modelDropdownMenu.appendChild(menuItem);
            }

            // Render Cotton Suite
            if (cottonModels.length > 0) {
                const cottonHeader = document.createElement("div");
                cottonHeader.className = "model-group-header";
                cottonHeader.innerHTML = `<i class="fa-solid fa-leaf"></i> Cotton Weed Suite (Vision Mamba)`;
                modelDropdownMenu.appendChild(cottonHeader);
                cottonModels.forEach(renderModelItem);
            }

            // Render Corn Suite
            if (cornModels.length > 0) {
                const cornHeader = document.createElement("div");
                cornHeader.className = "model-group-header";
                cornHeader.innerHTML = `<i class="fa-solid fa-seedling"></i> Corn Weed Suite (CBAM Attention)`;
                modelDropdownMenu.appendChild(cornHeader);
                cornModels.forEach(renderModelItem);
            }

            // Set indicators to online active status
            systemStatusIndicator.querySelector(".status-dot").className = "status-dot pulsing";
            systemStatusIndicator.querySelector(".status-text").textContent = "Vision Core Online";
        } catch (error) {
            console.error("Failed to load model registry:", error);
            activeModelName.textContent = "Offline Backup Mode";
            systemStatusIndicator.querySelector(".status-dot").className = "status-dot";
            systemStatusIndicator.querySelector(".status-text").textContent = "Offline Mode";
        }
    }

    // ============================================================
    // ATTACHMENT CONTROLLERS (Image Selection)
    // ============================================================
    attachFileBtn.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            loadSpecimenImage(e.target.files[0]);
        }
    });

    function loadSpecimenImage(file) {
        if (!file.type.startsWith("image/")) {
            alert("Specimen Error: Only leaf/crop images are acceptable.");
            return;
        }

        activeImageFile = file;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            // Load Thumbnail image
            inputImageThumbnail.src = e.target.result;
            selectedFilename.textContent = file.name.length > 25 ? file.name.substring(0, 22) + "..." : file.name;
            selectedImageBar.style.display = "flex";
            
            // Activate Chat inputs
            chatTextInput.value = `Image attached: ${file.name}`;
            sendDiagnosisBtn.disabled = false;
            sendDiagnosisBtn.classList.add("active");
            
            // Turn off camera overlay if running
            closeCamera();
        };
        reader.readAsDataURL(file);
    }

    clearSelectedImage.addEventListener("click", (e) => {
        e.stopPropagation();
        resetInputPill();
    });

    function resetInputPill() {
        activeImageFile = null;
        inputImageThumbnail.src = "";
        selectedImageBar.style.display = "none";
        chatTextInput.value = "";
        fileInput.value = "";
        sendDiagnosisBtn.disabled = true;
        sendDiagnosisBtn.classList.remove("active");
    }

    // ============================================================
    // CAMERA SNAPSHOT API HOOKS
    // ============================================================
    async function openCamera() {
        if (cameraOverlay.style.display === "flex") {
            closeCamera();
            return;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Camera Access Note: Direct webcam streaming requires localhost (http://localhost:8000) or HTTPS. Opening file selector so you can choose or take a photo!");
            fileInput.value = "";
            fileInput.click();
            return;
        }
        
        try {
            cameraStreamTrack = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: 800, height: 600 },
                audio: false
            });
            cameraStream.srcObject = cameraStreamTrack;
            cameraOverlay.style.display = "flex";
            resetInputPill(); // Remove any attached file
        } catch (error) {
            console.error("Camera hook failed:", error);
            alert("Camera Note: Could not acquire camera stream. Opening file selector so you can select a specimen image.");
            fileInput.value = "";
            fileInput.click();
        }
    }

    openCameraBtn.addEventListener("click", openCamera);

    function closeCamera() {
        if (cameraStreamTrack) {
            const tracks = cameraStreamTrack.getTracks();
            tracks.forEach(t => t.stop());
            cameraStream.srcObject = null;
            cameraStreamTrack = null;
        }
        cameraOverlay.style.display = "none";
    }

    closeCameraBtn.addEventListener("click", closeCamera);

    captureSnapshotBtn.addEventListener("click", () => {
        if (!cameraStreamTrack) return;
        
        cameraCanvas.width = cameraStream.videoWidth;
        cameraCanvas.height = cameraStream.videoHeight;
        
        const ctx = cameraCanvas.getContext("2d");
        ctx.drawImage(cameraStream, 0, 0, cameraCanvas.width, cameraCanvas.height);
        
        cameraCanvas.toBlob(blob => {
            const capturedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            loadSpecimenImage(capturedFile);
        }, "image/jpeg", 0.95);
    });

    // ============================================================
    // CONVERSATION ENGINE (CHAT MESSAGING DIALOGUE)
    // ============================================================
    sendDiagnosisBtn.addEventListener("click", triggerScanDiagnosis);

    // Press Enter to send if image is selected
    document.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            if (activeImageFile && !sendDiagnosisBtn.disabled) {
                e.preventDefault();
                triggerScanDiagnosis();
            }
        }
    });

    // Press Ctrl+K (or Cmd+K) to trigger file upload popup
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            fileInput.click();
        }
    });

    async function triggerScanDiagnosis() {
        if (!activeImageFile) return;

        const fileToSubmit = activeImageFile;
        const modelToRun = selectedModelId;
        const imageUrlData = inputImageThumbnail.src;

        // Reset Pill instantly so user feels ChatGPT responsive feedback
        resetInputPill();

        // 1. Swap screen states (Welcome -> Thread)
        welcomeScreen.style.display = "none";
        messageThread.style.display = "flex";

        // 2. Append User Chat Bubble
        appendUserMessage(imageUrlData, fileToSubmit.name);
        
        // 3. Append AI Chat Bubble with typing animation dots
        const typingBubbleId = appendAITypingIndicator(modelToRun);
        scrollFeedToBottom();

        // Lock trigger controls & show sleek analyzing spinner
        sendDiagnosisBtn.disabled = true;
        sendDiagnosisBtn.classList.add("analyzing");
        sendDiagnosisBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
        modelDropdownBtn.classList.add("pointer-none");

        // Submit to Flask REST API
        const formData = new FormData();
        formData.append("image", fileToSubmit);
        formData.append("model", modelToRun);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Inference error");
            const result = await response.json();
            
            // Remove typing bubble and render final diagnostic card bubble!
            removeMessageBubble(typingBubbleId);
            
            if (result.success) {
                appendAIDiagnosisReport(result, modelToRun);
                saveDiagnosisToHistory(result, modelToRun, imageUrlData);
            } else if (result.is_leaf === false) {
                appendSpecimenRejectionMessage(result, modelToRun);
            } else {
                appendSimpleAIMessage(`Error running TensorFlow inference: ${result.error}`, modelToRun);
            }
        } catch (error) {
            console.error("API call failed:", error);
            removeMessageBubble(typingBubbleId);
            appendSimpleAIMessage("Diagnosis Failure: The neural core backend is currently offline. Ensure `app.py` is active in the background.", modelToRun);
        } finally {
            // Unlock model controls & reset send button
            sendDiagnosisBtn.classList.remove("analyzing");
            sendDiagnosisBtn.innerHTML = '<i class="fa-solid fa-arrow-up"></i>';
            if (activeImageFile) {
                sendDiagnosisBtn.disabled = false;
                sendDiagnosisBtn.classList.add("active");
            } else {
                sendDiagnosisBtn.disabled = true;
                sendDiagnosisBtn.classList.remove("active");
            }
            modelDropdownBtn.classList.remove("pointer-none");
            scrollFeedToBottom();
        }
    }

    function scrollFeedToBottom() {
        chatFeed.scrollTop = chatFeed.scrollHeight;
    }

    // APPEND BUBBLES METHODS
    function appendUserMessage(imgUrl, filename) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble user";
        
        const userQuery = (chatTextInput.value || "").trim();
        const displayQuery = userQuery.length > 0 ? userQuery : "Please diagnose this crop specimen leaf for invasive weeds.";

        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-user"></i></div>
            <div class="bubble-content">
                <div class="bubble-text-wrapper">
                    <div class="bubble-text">${displayQuery}</div>
                </div>
                <div class="bubble-attachment-preview">
                    <img src="${imgUrl}" alt="Attachment Specimen">
                </div>
            </div>
        `;
        
        messageThread.appendChild(bubble);
        chatTextInput.value = "";
    }

    function appendAITypingIndicator(modelId) {
        const bubbleId = "typing-" + Date.now();
        const bubble = document.createElement("div");
        bubble.className = "message-bubble ai";
        bubble.id = bubbleId;
        
        const isCotton = modelId.toLowerCase().includes("cotton");
        const statusMsg = isCotton
            ? `Running Vision Mamba state-space blocks on ${modelId}...`
            : `Running CBAM attention gates on ${modelId}...`;

        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-leaf"></i></div>
            <div class="bubble-content">
                <div class="bubble-text" style="color: var(--text-muted); font-size: 11px;">
                    ${statusMsg}
                </div>
                <div class="typing-container">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        
        messageThread.appendChild(bubble);
        return bubbleId;
    }

    function removeMessageBubble(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function appendSimpleAIMessage(text, modelId) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble ai";
        
        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-leaf"></i></div>
            <div class="bubble-content">
                <span class="report-badge" style="width: fit-content; margin-bottom: 4px;">${modelId}</span>
                <div class="bubble-text">${text}</div>
            </div>
        `;
        
        messageThread.appendChild(bubble);
    }

    function appendSpecimenRejectionMessage(result, modelId) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble ai";

        const msgText = (result.message || "").toLowerCase();
        let iconHtml = '<i class="fa-solid fa-ban"></i>';
        let cardTitle = "Specimen Rejected: Non-Leaf Image";

        if (msgText.includes("fruit") || msgText.includes("apple") || msgText.includes("citrus") || msgText.includes("produce") || msgText.includes("ear") || msgText.includes("cob")) {
            iconHtml = '<i class="fa-solid fa-apple-whole"></i>';
            cardTitle = "Specimen Rejected: Fruit / Horticultural Produce";
        } else if (msgText.includes("animal") || msgText.includes("wildlife")) {
            iconHtml = '<i class="fa-solid fa-paw"></i>';
            cardTitle = "Specimen Rejected: Animal / Non-Plant Object";
        } else if (msgText.includes("culinary") || msgText.includes("food")) {
            iconHtml = '<i class="fa-solid fa-utensils"></i>';
            cardTitle = "Specimen Rejected: Culinary / Food Item";
        } else if (msgText.includes("flower") || msgText.includes("blossom")) {
            iconHtml = '<i class="fa-solid fa-spa"></i>';
            cardTitle = "Specimen Rejected: Ornamental Flower Blossom";
        }

        bubble.innerHTML = `
            <div class="bubble-avatar" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="bubble-content">
                <div class="nested-report-card specimen-rejection-card">
                    <div class="nested-report-header" style="border-bottom: 1px solid rgba(239, 68, 68, 0.25);">
                        <h3 style="color: #ef4444;"><i class="fa-solid fa-circle-xmark"></i> ${cardTitle}</h3>
                        <span class="report-badge" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);">${modelId} Gate</span>
                    </div>
                    
                    <div class="nested-report-body" style="gap: 12px; padding: 16px;">
                        <div class="rejection-alert-banner">
                            ${iconHtml}
                            <div>
                                <strong>${result.error || "Non-Leaf Specimen Detected"}</strong>
                                <p>${result.message || "No plant foliage or chlorophyll characteristics detected in the uploaded specimen."}</p>
                            </div>
                        </div>

                        <div class="nested-advisor-box" style="background: rgba(239, 68, 68, 0.05); border-left: 3px solid #ef4444; margin-top: 10px;">
                            <div class="nested-advisor-item">
                                <h4 style="color: #ef4444;">Agronomy Specimen Constraint:</h4>
                                <p>${modelId} and AgriShield are scientific deep learning networks trained exclusively on authentic vegetative crop leaves (Cotton / Gossypium hirsutum &amp; Corn / Zea mays) and invasive field weeds. Fruits (apples, citrus, berries), harvested corn cobs, flower bouquets, food, animals, or digital art are outside the training distribution and strictly rejected to prevent false positive classifications.</p>
                            </div>
                            <div class="nested-advisor-item" style="margin-top: 8px;">
                                <h4 style="color: var(--accent-color);">Required Action:</h4>
                                <p>${result.suggestion || "Please upload an authentic photograph of a crop leaf blade (Cotton or Corn) or an invasive weed specimen."}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        messageThread.appendChild(bubble);
    }

    function appendAIDiagnosisReport(result, modelId) {
        const bubble = document.createElement("div");
        bubble.className = "message-bubble ai";
        
        const className = result.class_name;
        const dbInfo = BOTANICAL_DATABASE[className] || {
            commonName: className,
            scientificName: className,
            risk: "medium",
            description: "Agricultural specimen identified via deep learning pattern matching.",
            prevention: "Implement standard integrated weed management (IWM) scouting and prevention."
        };
        const modeLabel = result.mode === "trained_deep_learning" ? "Trained Graph Core" : "Vision Engine";
        const badgeClass = result.mode === "trained_deep_learning" ? "" : "warning";
        const cropName = result.crop || (modelId.toLowerCase().includes("cotton") ? "Cotton" : "Corn");
        const archName = result.architecture || (cropName === "Cotton" ? "Vision Mamba" : "CBAM");
        
        // Progress ring offset calculations
        const perimeter = 2 * Math.PI * 40;
        const dashOffset = perimeter - (perimeter * result.confidence) / 100;
        
        // Softmax probability breakdown sorted by confidence descending
        const sortedProbs = Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]);
        let probBarsHtml = "";
        sortedProbs.forEach(([name, prob]) => {
            probBarsHtml += `
                <div class="nested-bar-row">
                    <div class="nested-bar-labels">
                        <span class="nested-bar-name">${name}</span>
                        <span class="nested-bar-val">${prob}%</span>
                    </div>
                    <div class="nested-bar-bg">
                        <div class="nested-bar-fill" style="width: ${prob}%"></div>
                    </div>
                </div>
            `;
        });

        bubble.innerHTML = `
            <div class="bubble-avatar" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(56, 189, 248, 0.2)); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.35);"><i class="fa-solid fa-wand-magic-sparkles"></i></div>
            <div class="bubble-content">
                <div class="ai-thought-pill">
                    <i class="fa-solid fa-brain"></i>
                    <span>Neural Attention Core &bull; Verified with ${modelId} (${archName}) &bull; ${result.confidence}% confidence</span>
                </div>
                
                <div class="bubble-text">Neural scanning complete. Attention mapping identifies localized crop features. The diagnostic report sheet is detailed below:</div>
                
                <!-- Nested Report Card inside bubble -->
                <div class="nested-report-card">
                    <div class="nested-report-header">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <img src="/igu_seal.png" alt="IGU Seal" style="width: 22px; height: 22px; border-radius: 50%; object-fit: contain; box-shadow: 0 0 8px rgba(16, 185, 129, 0.2);">
                            <h3>Agri-Scan Diagnostics</h3>
                        </div>
                        <div style="display: flex; gap: 6px; align-items: center;">
                            <span class="report-badge" style="background: rgba(16, 185, 129, 0.15); color: var(--accent-color); border: 1px solid rgba(16, 185, 129, 0.3);">${cropName} • ${archName}</span>
                            <span class="report-badge ${badgeClass}">${modeLabel}</span>
                        </div>
                    </div>
                    
                    <div class="nested-report-body">
                        <!-- Ring and classification details -->
                        <div class="nested-metric-row">
                            <div class="nested-ring-container">
                                <svg class="nested-progress-ring" width="90" height="90">
                                    <circle class="nested-progress-ring-bg" stroke="rgba(255,255,255,0.06)" stroke-width="6" fill="transparent" r="40" cx="45" cy="45"/>
                                    <circle class="nested-progress-ring-bar" stroke-width="6" fill="transparent" r="40" cx="45" cy="45" stroke-dasharray="${perimeter}" stroke-dashoffset="${dashOffset}"/>
                                </svg>
                                <div class="nested-ring-text">
                                    <span class="nested-percentage-num">${result.confidence}%</span>
                                    <span class="nested-percentage-lbl">Accuracy</span>
                                </div>
                            </div>
                            
                            <div class="nested-info-box">
                                <span class="nested-label-heading">CLASSIFIED SPECIES</span>
                                <h3 class="nested-entity-title">${dbInfo.commonName}</h3>
                                <span class="nested-botanical-name">${dbInfo.scientificName}</span>
                                <span class="nested-risk-badge ${dbInfo.risk}"><i class="fa-solid fa-triangle-exclamation"></i> Risk: ${dbInfo.risk}</span>
                            </div>
                        </div>
                        
                        <!-- Softmax progress bars -->
                        <div class="nested-probabilities-container">
                            <span class="nested-probabilities-title">Softmax Probability breakdown</span>
                            ${probBarsHtml}
                        </div>
                        
                        <!-- Botanical advice sheet -->
                        <div class="nested-advisor-box">
                            <div class="nested-advisor-item">
                                <h4>Botanical Overview:</h4>
                                <p>${dbInfo.description}</p>
                            </div>
                            <div class="nested-advisor-item" style="margin-top: 6px;">
                                <h4>Prevention &amp; Sprays Plan:</h4>
                                <p>${dbInfo.prevention}</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Actions (ChatGPT & Gemini Style) -->
                    <div class="nested-report-footer">
                        <button class="action-pill-btn copy-report-btn" title="Copy diagnosis to clipboard">
                            <i class="fa-regular fa-copy"></i> Copy Summary
                        </button>
                        <button class="action-pill-btn" onclick="window.print()" title="Print diagnostic sheet">
                            <i class="fa-solid fa-print"></i> Print Sheet
                        </button>
                        <button class="action-pill-btn feedback-thumb-btn" data-dir="up" title="Helpful analysis">
                            <i class="fa-regular fa-thumbs-up"></i>
                        </button>
                        <button class="action-pill-btn feedback-thumb-btn" data-dir="down" title="Report issue">
                            <i class="fa-regular fa-thumbs-down"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        // Wire up copy button
        const copyBtn = bubble.querySelector(".copy-report-btn");
        if (copyBtn) {
            copyBtn.addEventListener("click", () => {
                const reportText = `AgriShield AI Diagnosis Report (IGU Meerpur, Rewari)\nSpecimen: ${dbInfo.commonName} (${dbInfo.scientificName})\nCrop Group: ${cropName} | Model: ${modelId} (${archName})\nConfidence: ${result.confidence}%\nRisk Level: ${dbInfo.risk.toUpperCase()}\nManagement: ${dbInfo.prevention}`;
                navigator.clipboard.writeText(reportText).then(() => {
                    copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
                    copyBtn.classList.add("active");
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy Summary';
                        copyBtn.classList.remove("active");
                    }, 2000);
                });
            });
        }

        // Wire up feedback buttons
        bubble.querySelectorAll(".feedback-thumb-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const isAlready = btn.classList.contains("active");
                bubble.querySelectorAll(".feedback-thumb-btn").forEach(b => b.classList.remove("active"));
                if (!isAlready) btn.classList.add("active");
            });
        });

        messageThread.appendChild(bubble);
    }

    // ============================================================
    // CHAT HISTORY STORAGE & LOAD MECHANICS
    // ============================================================
    function saveDiagnosisToHistory(result, modelId, imgUrl) {
        const scanItem = {
            id: "scan-" + Date.now(),
            modelId: modelId,
            className: result.class_name,
            confidence: result.confidence,
            result: result,
            imageUrl: imgUrl
        };
        
        diagnosisHistoryList.unshift(scanItem);
        renderHistoryList();
    }

    function renderHistoryList() {
        if (diagnosisHistoryList.length === 0) {
            historyScansList.innerHTML = `
                <div class="history-item empty">
                    <i class="fa-solid fa-message-medical"></i> No recent scans
                </div>
            `;
            return;
        }
        
        historyScansList.innerHTML = "";
        diagnosisHistoryList.forEach(item => {
            const historyItem = document.createElement("div");
            historyItem.className = "history-item";
            historyItem.dataset.id = item.id;
            
            const dbInfo = BOTANICAL_DATABASE[item.className] || { commonName: item.className };
            const displayTitle = (dbInfo.commonName || item.className).split(" ")[0];
            
            historyItem.innerHTML = `
                <i class="fa-solid fa-file-invoice-leaf" style="color: var(--accent-color);"></i>
                <span>${displayTitle} (${item.confidence}%)</span>
            `;
            
            historyItem.addEventListener("click", () => {
                document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));
                historyItem.classList.add("active");
                loadScanFromHistory(item);
            });
            
            historyScansList.appendChild(historyItem);
        });
    }

    function loadScanFromHistory(item) {
        // Toggle view states
        welcomeScreen.style.display = "none";
        messageThread.style.display = "flex";
        messageThread.innerHTML = ""; // Clear active conversation
        
        // Reload bubbles
        appendUserMessage(item.imageUrl, "specimen.jpg");
        appendAIDiagnosisReport(item.result, item.modelId);
        scrollFeedToBottom();
    }

    // New Chat handler
    newDiagnosisBtn.addEventListener("click", () => {
        messageThread.style.display = "none";
        messageThread.innerHTML = "";
        welcomeScreen.style.display = "flex";
        resetInputPill();
        closeCamera();
        
        document.querySelectorAll(".history-item").forEach(i => i.classList.remove("active"));
    });

    // Suggestion card clicks (Gemini / ChatGPT smart actions)
    document.querySelectorAll(".suggestion-card").forEach(card => {
        card.addEventListener("click", () => {
            const action = card.dataset.action;
            if (action === "upload-cotton") {
                setActiveModel("InceptionV3_Cotton_Mamba");
                chatTextInput.value = "Diagnose Cotton leaf specimen with Vision Mamba...";
                fileInput.value = "";
                fileInput.click();
            } else if (action === "upload-corn") {
                setActiveModel("Untitled65");
                chatTextInput.value = "Diagnose Corn (Maize) leaf specimen with CBAM attention...";
                fileInput.value = "";
                fileInput.click();
            } else if (action === "upload-weed") {
                setActiveModel("InceptionV3_Cotton_Mamba");
                chatTextInput.value = "Identify invasive weed specimen and management advice...";
                fileInput.value = "";
                fileInput.click();
            } else if (action === "open-camera") {
                openCamera();
            } else {
                fileInput.value = "";
                fileInput.click();
            }
        });
    });

    // Leaderboard Modal Controls
    const leaderboardModal = document.getElementById("leaderboard-modal");
    const openBenchmarksBtn = document.getElementById("open-benchmarks-btn");
    const openUnivInfo = document.getElementById("open-univ-info");
    const closeLeaderboardBtn = document.getElementById("close-leaderboard-btn");

    function openLeaderboard() {
        if (leaderboardModal) leaderboardModal.style.display = "flex";
    }

    function closeLeaderboard() {
        if (leaderboardModal) leaderboardModal.style.display = "none";
    }

    if (openBenchmarksBtn) openBenchmarksBtn.addEventListener("click", openLeaderboard);
    if (openUnivInfo) openUnivInfo.addEventListener("click", openLeaderboard);
    if (closeLeaderboardBtn) closeLeaderboardBtn.addEventListener("click", closeLeaderboard);
    if (leaderboardModal) {
        leaderboardModal.addEventListener("click", (e) => {
            if (e.target === leaderboardModal) closeLeaderboard();
        });
    }

    // ============================================================
    // DRAG AND DROP CAPABILITIES
    // ============================================================
    const chatViewport = document.getElementById("chat-viewport");
    const dragDropOverlay = document.getElementById("drag-drop-overlay");

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        chatViewport.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        chatViewport.addEventListener(eventName, showDragOverlay, false);
    });

    function showDragOverlay() {
        dragDropOverlay.style.display = "flex";
    }

    // Remove highlight when item is dragged away
    dragDropOverlay.addEventListener('dragleave', hideDragOverlay, false);

    function hideDragOverlay(e) {
        const rect = dragDropOverlay.getBoundingClientRect();
        if (e.clientX < rect.left || e.clientX >= rect.right || e.clientY < rect.top || e.clientY >= rect.bottom) {
            dragDropOverlay.style.display = "none";
        }
    }

    // Handle dropped files
    chatViewport.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        dragDropOverlay.style.display = "none";
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            loadSpecimenImage(files[0]);
        }
    }

    // Welcome Screen Dropzone Click & Drag events
    if (welcomeDropzone) {
        welcomeDropzone.addEventListener("click", () => {
            fileInput.click();
        });

        welcomeDropzone.addEventListener("dragenter", preventDefaults, false);
        welcomeDropzone.addEventListener("dragover", (e) => {
            preventDefaults(e);
            welcomeDropzone.classList.add("dragover");
        }, false);

        welcomeDropzone.addEventListener("dragleave", (e) => {
            preventDefaults(e);
            welcomeDropzone.classList.remove("dragover");
        }, false);

        welcomeDropzone.addEventListener("drop", (e) => {
            preventDefaults(e);
            welcomeDropzone.classList.remove("dragover");
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                loadSpecimenImage(files[0]);
            }
        }, false);
    }

    // ============================================================
    // MOBILE TOUCH GESTURES (SWIPES FOR SIDEBAR)
    // ============================================================
    let touchStartX = 0;
    let touchStartY = 0;
    let touchEndX = 0;
    let touchEndY = 0;

    chatViewport.addEventListener("touchstart", (e) => {
        touchStartX = e.changedTouches[0].screenX;
        touchStartY = e.changedTouches[0].screenY;
    }, { passive: true });

    chatViewport.addEventListener("touchend", (e) => {
        touchEndX = e.changedTouches[0].screenX;
        touchEndY = e.changedTouches[0].screenY;
        handleSwipeGesture();
    }, { passive: true });

    function handleSwipeGesture() {
        const diffX = touchEndX - touchStartX;
        const diffY = touchEndY - touchStartY;
        
        // Horizontal swipe check: distance > 80px and angle is mostly horizontal
        if (Math.abs(diffX) > 80 && Math.abs(diffY) < 60) {
            if (diffX > 0) {
                // Swipe Left-to-Right: Open sidebar
                openSidebar();
            } else {
                // Swipe Right-to-Left: Close sidebar
                closeSidebar();
            }
        }
    }

    // ============================================================
    // APPLICATION INITIALIZATION
    // ============================================================
    loadModelsRegistry();
});
