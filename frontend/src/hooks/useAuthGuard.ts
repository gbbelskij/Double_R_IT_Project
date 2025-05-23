import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import axios from "axios";

const getTokenFromCookie = (): string | null => {
  const match = document.cookie.match(/(?:^|;\s*)token=([^;]*)/);
  return match ? decodeURIComponent(match[1]) : null;
};

export const useAuthGuard = (): boolean => {
  const [isChecking, setIsChecking] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      const token = getTokenFromCookie();

      if (!token) {
        navigate("/login");
        return;
      }

      try {
        await axios.get("/api/verify/", {
          withCredentials: true,
        });
      } catch (err) {
        document.cookie = "token=; Max-Age=0; path=/";
        navigate("/login");

        if (axios.isAxiosError(err)) {
          const message =
            err.response?.data?.message || err.message || "Auth error";
          console.error("Auth verification error:", message);
        } else {
          console.error("Unknown error during auth verification:", err);
        }
      } finally {
        setIsChecking(false);
      }
    };

    checkAuth();
  }, [navigate]);

  return isChecking;
};
