import { useNavigate } from "react-router";

import { FaRegUserCircle } from "react-icons/fa";

import { useWindowSize } from "@hooks/useWindowSize";

import Logo from "@components/Logo/Logo";
import Button from "@components/Button/Button";

import classes from "./Header.module.css";

const Header: React.FC = () => {
  const navigate = useNavigate();
  const handleProfileClick = () => navigate("/profile");

  const { width: windowWidth } = useWindowSize();

  return (
    <header className={classes.Header}>
      <Logo hasText />

      {windowWidth >= 440 ? (
        <Button
          color="dim"
          rightIcon={<FaRegUserCircle size={18} />}
          onClick={handleProfileClick}
        >
          Профиль
        </Button>
      ) : (
        <FaRegUserCircle
          size={30}
          onClick={handleProfileClick}
          className={classes.HeaderProfileButton}
        />
      )}
    </header>
  );
};

export default Header;
