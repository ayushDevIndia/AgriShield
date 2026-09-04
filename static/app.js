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
    let selectedModelId = "LNet";
    let activeImageFile = null;
    let cameraStreamTrack = null;
    let diagnosisHistoryList = [];
    let activeModelList = [];

    // Botanical preventive guide database
    const BOTANICAL_DATABASE = {
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

    const CLASS_NAMES = ["bluegrass", "chenopodium album", "cirsium setosum", "corn", "sedge"];

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

    async function loadModelsRegistry() {
        try {
            const response = await fetch("/api/models");
            if (!response.ok) throw new Error("Registry server down");
            activeModelList = await response.json();
            
            // Set first model active as trigger label
            const defaultModel = activeModelList.find(m => m.id === selectedModelId);
            if (defaultModel) {
                activeModelName.textContent = defaultModel.name;
            }

            modelDropdownMenu.innerHTML = "";
            activeModelList.forEach(model => {
                const isActive = model.id === selectedModelId ? "active" : "";
                const menuItem = document.createElement("div");
                menuItem.className = `model-menu-item ${isActive}`;
                menuItem.dataset.id = model.id;
                
                menuItem.innerHTML = `
                    <div class="model-menu-item-title">${model.name}</div>
                    <div class="model-menu-item-desc">Params: ${model.params} | Acc: ${model.accuracy}</div>
                `;
                
                menuItem.addEventListener("click", () => {
                    selectedModelId = model.id;
                    activeModelName.textContent = model.name;
                    document.querySelectorAll(".model-menu-item").forEach(item => item.classList.remove("active"));
                    menuItem.classList.add("active");
                });
                
                modelDropdownMenu.appendChild(menuItem);
            });

            // Set indicators to online active status
            systemStatusIndicator.querySelector(".status-dot").className = "status-dot pulsing";
            systemStatusIndicator.querySelector(".status-text").textContent = "Connected";
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
    openCameraBtn.addEventListener("click", async () => {
        if (cameraOverlay.style.display === "flex") {
            closeCamera();
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
            alert("System Camera Alert: Could not acquire stream. Ensure camera permissions are enabled.");
        }
    });

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

        // Lock trigger controls
        sendDiagnosisBtn.disabled = true;
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
            // Unlock model controls
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
        
        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-user"></i></div>
            <div class="bubble-content">
                <div class="bubble-text-wrapper">
                    <div class="bubble-text">Please diagnose this crop specimen leaf.</div>
                </div>
                <div class="bubble-attachment-preview">
                    <img src="${imgUrl}" alt="Attachment Specimen">
                </div>
            </div>
        `;
        
        messageThread.appendChild(bubble);
    }

    function appendAITypingIndicator(modelId) {
        const bubbleId = "typing-" + Date.now();
        const bubble = document.createElement("div");
        bubble.className = "message-bubble ai";
        bubble.id = bubbleId;
        
        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-leaf"></i></div>
            <div class="bubble-content">
                <div class="bubble-text" style="color: var(--text-muted); font-size: 11px;">
                    Running CBAM attention gates on ${modelId} layers...
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

        bubble.innerHTML = `
            <div class="bubble-avatar" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i></div>
            <div class="bubble-content">
                <div class="nested-report-card specimen-rejection-card">
                    <div class="nested-report-header" style="border-bottom: 1px solid rgba(239, 68, 68, 0.25);">
                        <h3 style="color: #ef4444;"><i class="fa-solid fa-circle-xmark"></i> Specimen Rejected: Non-Leaf Image</h3>
                        <span class="report-badge" style="background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);">${modelId} Gate</span>
                    </div>
                    
                    <div class="nested-report-body" style="gap: 12px; padding: 16px;">
                        <div class="rejection-alert-banner">
                            <i class="fa-solid fa-ban"></i>
                            <div>
                                <strong>${result.error || "Non-Leaf Specimen Detected"}</strong>
                                <p>${result.message || "No plant foliage or chlorophyll characteristics detected in the uploaded specimen."}</p>
                            </div>
                        </div>

                        <div class="nested-advisor-box" style="background: rgba(239, 68, 68, 0.05); border-left: 3px solid #ef4444; margin-top: 10px;">
                            <div class="nested-advisor-item">
                                <h4 style="color: #ef4444;">Agronomy Specimen Constraint:</h4>
                                <p>${modelId} and AgriShield are trained exclusively to diagnose agricultural crop specimens (Corn / Zea mays) and invasive field weeds (Bluegrass, Chenopodium, Cirsium, Sedge). Images of humans, animals, objects, rooms, or documents are rejected to maintain strict scientific accuracy.</p>
                            </div>
                            <div class="nested-advisor-item" style="margin-top: 8px;">
                                <h4 style="color: var(--accent-color);">Required Action:</h4>
                                <p>${result.suggestion || "Please upload an authentic photograph of a crop leaf or weed specimen."}</p>
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
        const dbInfo = BOTANICAL_DATABASE[className];
        const modeLabel = result.mode === "trained_deep_learning" ? "Trained Graph Core" : "Simulation Engine";
        const badgeClass = result.mode === "trained_deep_learning" ? "" : "warning";
        
        // Progress ring offset calculations
        const perimeter = 2 * Math.PI * 40;
        const dashOffset = perimeter - (perimeter * result.confidence) / 100;
        
        // Probability bars HTML
        let probBarsHtml = "";
        CLASS_NAMES.forEach(c => {
            const prob = result.probabilities[c] || 0;
            probBarsHtml += `
                <div class="nested-bar-row">
                    <div class="nested-bar-labels">
                        <span class="nested-bar-name">${c}</span>
                        <span class="nested-bar-val">${prob}%</span>
                    </div>
                    <div class="nested-bar-bg">
                        <div class="nested-bar-fill" style="width: ${prob}%"></div>
                    </div>
                </div>
            `;
        });

        bubble.innerHTML = `
            <div class="bubble-avatar"><i class="fa-solid fa-leaf"></i></div>
            <div class="bubble-content">
                <div class="bubble-text">Neural scanning complete. Attention mapping identifies localized crop features. The diagnostic report sheet is detailed below:</div>
                
                <!-- Nested Report Card inside bubble -->
                <div class="nested-report-card">
                    <div class="nested-report-header">
                        <h3><i class="fa-solid fa-microchip"></i> Agri-Scan Diagnostics</h3>
                        <span class="report-badge ${badgeClass}">${modeLabel}</span>
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
                                <h4>Prevention & Sprays Plan:</h4>
                                <p>${dbInfo.prevention}</p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Actions -->
                    <div class="nested-report-footer">
                        <button class="nested-print-btn" onclick="window.print()"><i class="fa-solid fa-print"></i> Print Diagnosis Sheet</button>
                    </div>
                </div>
            </div>
        `;
        
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
            
            const dbInfo = BOTANICAL_DATABASE[item.className];
            const displayTitle = dbInfo.commonName.split(" ")[0]; // First word of common name
            
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

    // Suggesion card clicks
    document.querySelectorAll(".suggestion-card").forEach(card => {
        card.addEventListener("click", () => {
            // Trigger browser select file dialog directly!
            fileInput.click();
        });
    });

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
