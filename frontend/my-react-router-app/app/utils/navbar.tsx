import { useState } from "react";
import { Link, NavLink } from "react-router";

export type NavItem = { to: string; label: string };

type NavBarProps = {
  brand?: string;
  logoSrc?: string;
  items?: NavItem[];
};

const defaultItems: NavItem[] = [
  { to: "/", label: "Home" },
  { to: "/exercises", label: "Exercises" },
];

export default function NavBar({
  brand = "E-Learning",
  logoSrc = "https://flowbite.com/docs/images/logo.svg",
  items = defaultItems,
}: NavBarProps) {
  const [open, setOpen] = useState(false);

  return (
    <nav className="bg-neutral-primary fixed w-full z-20 top-0 start-0 border-b border-default">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        <Link to="/" className="flex items-center space-x-3 rtl:space-x-reverse">
          <img src={logoSrc} className="h-7" alt={`${brand} Logo`} />
          <span className="self-center text-xl text-heading font-semibold whitespace-nowrap">{brand}</span>
        </Link>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-body rounded-base md:hidden hover:bg-neutral-secondary-soft hover:text-heading focus:outline-none focus:ring-2 focus:ring-neutral-tertiary"
          aria-controls="navbar-default"
          aria-expanded={open}
        >
          <span className="sr-only">Open main menu</span>
          <svg className="w-6 h-6" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
            <path stroke="currentColor" strokeLinecap="round" strokeWidth="2" d="M5 7h14M5 12h14M5 17h14" />
          </svg>
        </button>
        <div className={`${open ? "block" : "hidden"} w-full md:block md:w-auto`} id="navbar-default">
          <ul className="font-medium flex flex-col p-4 md:p-0 mt-4 border border-default rounded-base bg-neutral-secondary-soft md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:border-0 md:bg-neutral-primary">
            {items.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end
                  onClick={() => setOpen(false)}
                  className={({ isActive }) =>
                    isActive
                      ? "block py-2 px-3 text-white bg-brand rounded md:bg-transparent md:text-fg-brand md:p-0"
                      : "block py-2 px-3 text-heading rounded hover:bg-neutral-tertiary md:hover:bg-transparent md:border-0 md:hover:text-fg-brand md:p-0"
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </nav>
  );
}
