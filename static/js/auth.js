const authButtons = document.querySelectorAll(".btn-auth");
const authModal = document.getElementById("authModal");
const authClose = document.getElementById("closeAuthModal");
const loginForm = document.getElementById("loginForm");
const loginMessage = document.getElementById("loginMessage");
const openRecoverButton = document.getElementById("openRecoverWindow");
const openSignupButton = document.getElementById("openSignupWindow");
const authStorageKey = "unionAuthUser";

function getStoredUser() {
  const raw = localStorage.getItem(authStorageKey);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw);
  } catch (error) {
    localStorage.removeItem(authStorageKey);
    return null;
  }
}

function setStoredUser(profile) {
  localStorage.setItem(authStorageKey, JSON.stringify(profile));
}

function clearStoredUser() {
  localStorage.removeItem(authStorageKey);
}

function renderAuthState() {
  const user = getStoredUser();

  authButtons.forEach((button) => {
    const parent = button.parentElement;
    if (!parent) {
      return;
    }

    let statusBox = parent.querySelector(".auth-status");

    if (user) {
      if (!statusBox) {
        statusBox = document.createElement("div");
        statusBox.className = "auth-status";

        const userName = document.createElement("span");
        userName.className = "auth-user";

        const logoutButton = document.createElement("button");
        logoutButton.type = "button";
        logoutButton.className = "auth-logout";
        logoutButton.textContent = "\uB85C\uADF8\uC544\uC6C3";

        statusBox.appendChild(userName);
        statusBox.appendChild(logoutButton);
        parent.replaceChild(statusBox, button);
      }

      statusBox.querySelector(".auth-user").textContent = `${user.name}\uB2D8`;
      statusBox.querySelector(".auth-logout").onclick = () => {
        clearStoredUser();
        closeAuthModal();
        renderAuthState();
      };
    } else if (statusBox) {
      parent.replaceChild(button, statusBox);
    }
  });
}

function openAuthModal(event) {
  if (event) {
    event.preventDefault();
  }

  if (getStoredUser()) {
    return;
  }

  authModal.classList.remove("auth-hidden");
}

function closeAuthModal() {
  authModal.classList.add("auth-hidden");
  loginMessage.textContent = "";
  loginMessage.className = "auth-message";
}

function openPopup(url, name) {
  window.open(url, name, "width=760,height=880,scrollbars=yes,resizable=yes");
}

authButtons.forEach((button) => {
  button.addEventListener("click", openAuthModal);
});

renderAuthState();

window.addEventListener("storage", (event) => {
  if (event.key === authStorageKey) {
    renderAuthState();
  }
});

if (authClose) {
  authClose.addEventListener("click", closeAuthModal);
}

if (authModal) {
  authModal.addEventListener("click", (event) => {
    if (event.target === authModal) {
      closeAuthModal();
    }
  });
}

if (openRecoverButton) {
  openRecoverButton.addEventListener("click", () => {
    openPopup("/recover.html", "unionRecoverWindow");
  });
}

if (openSignupButton) {
  openSignupButton.addEventListener("click", () => {
    openPopup("/signup.html", "unionSignupWindow");
  });
}

if (loginForm) {
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const payload = {
      username: document.getElementById("loginUsername").value.trim(),
      password: document.getElementById("loginPassword").value.trim()
    };

    try {
      const response = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || "\uB85C\uADF8\uC778\uC5D0 \uC2E4\uD328\uD588\uC2B5\uB2C8\uB2E4.");
      }

      setStoredUser(result.profile);
      loginMessage.textContent = `${result.message} (${result.profile.username})`;
      loginMessage.className = "auth-message success";
      alert(`${result.profile.name}\uB2D8\uC758 \uB85C\uADF8\uC778\uC5D0 \uC131\uACF5\uD588\uC2B5\uB2C8\uB2E4.`);
      loginForm.reset();
      closeAuthModal();
      renderAuthState();
    } catch (error) {
      loginMessage.textContent = error.message;
      loginMessage.className = "auth-message error";
    }
  });
}
